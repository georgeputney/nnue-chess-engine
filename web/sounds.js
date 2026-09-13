/*
 * Move sounds, synthesised rather than shipped.
 *
 * No audio files: nothing to download, nothing to license - the ready-made chess sound sets
 * are mostly GPL - and a piece landing on a board is not a complicated noise.
 *
 * It is modal synthesis, which is what makes it read as wood rather than as a beep:
 *   - a very short band-passed noise transient (under 8 ms) for the contact itself. A longer
 *     burst stops being a click and becomes a hiss, which was the first version's mistake.
 *   - three damped sine partials at inharmonic ratios for the body. Wood resonates at ratios
 *     that are not whole numbers, and the higher modes die first; one tone sounds synthetic
 *     however you shape it.
 *   - a little randomness per strike, because no two pieces land identically.
 *
 * The context is created on the first click, since browsers refuse one before a gesture, and
 * the preference is remembered per visitor.
 */

// Tuned dry and bright rather than woody: the contact carries the sound and the body barely
// rings, which is what a piece on a hard board actually does. An earlier pass sat at 210 Hz
// with a 120 ms tail and came out as a woodblock.
const MODES = [1, 2.42, 4.13];
const MODE_LEVEL = [1, 0.28, 0.09];
const MODE_DECAY = [1, 0.48, 0.26];

let audio = null;
let master = null;
// Off unless this visitor has turned it on. A page that starts making noise is a page people
// close, and a chess board is not obviously one that will.
let muted = true;
try {
  muted = localStorage.getItem("muted") !== "0";
} catch {
  muted = true;    // private windows and blocked storage both throw here
}

function context() {
  if (!audio) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    if (!Ctor) return null;
    audio = new Ctor();
    master = audio.createGain();
    master.gain.value = 0.55;
    master.connect(audio.destination);
  }
  if (audio.state === "suspended") audio.resume();
  return audio;
}

const vary = (x, amount) => x * (1 + (Math.random() * 2 - 1) * amount);

// The contact. Band-passed so it is a tick rather than a hiss, and short enough to be over
// before the body has finished its first cycle.
function transient(ctx, at, { level, centre, decay }) {
  const n = Math.max(1, Math.ceil(ctx.sampleRate * decay));
  const buffer = ctx.createBuffer(1, n, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < n; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / n) ** 2;

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  const band = ctx.createBiquadFilter();
  band.type = "bandpass";
  band.frequency.value = centre;
  band.Q.value = 0.9;
  const gain = ctx.createGain();
  gain.gain.value = level;

  source.connect(band).connect(gain).connect(master);
  source.start(at);
}

// One resonant mode: a sine with an exponential tail.
function mode(ctx, at, freq, level, decay) {
  const osc = ctx.createOscillator();
  osc.type = "sine";
  osc.frequency.setValueAtTime(freq, at);

  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0, at);
  gain.gain.linearRampToValueAtTime(level, at + 0.002);     // 2 ms attack, not a click of its own
  gain.gain.exponentialRampToValueAtTime(0.0001, at + decay);

  osc.connect(gain).connect(master);
  osc.start(at);
  osc.stop(at + decay + 0.02);
}

// Grit. Soft-clipping a noise burst turns a clean thud into a crunch, which is the thing that
// separates a capture from a move by character rather than by pitch.
function shaper(ctx, amount) {
  const curve = new Float32Array(1024);
  for (let i = 0; i < curve.length; i++) curve[i] = Math.tanh((i / 512 - 1) * amount);
  const node = ctx.createWaveShaper();
  node.curve = curve;
  return node;
}

// A crunch: longer than a contact, rough, and pushed through the shaper.
function crunch(ctx, at, { level, centre, decay }) {
  const n = Math.max(1, Math.ceil(ctx.sampleRate * decay));
  const buffer = ctx.createBuffer(1, n, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < n; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / n) ** 1.4;

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  const band = ctx.createBiquadFilter();
  band.type = "bandpass";
  band.frequency.value = centre;
  band.Q.value = 0.6;
  const gain = ctx.createGain();
  gain.gain.value = level;

  source.connect(shaper(ctx, 3.2)).connect(band).connect(gain).connect(master);
  source.start(at);
}

// A slide: noise swept upward through a band-pass, which is a rook travelling, not a piece
// landing. Castling is the one move that is two pieces moving, and it should sound like it.
function sweep(ctx, at, { level, from, to, decay }) {
  const n = Math.max(1, Math.ceil(ctx.sampleRate * decay));
  const buffer = ctx.createBuffer(1, n, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < n; i++) data[i] = (Math.random() * 2 - 1) * Math.sin((i / n) * Math.PI);

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  const band = ctx.createBiquadFilter();
  band.type = "bandpass";
  band.Q.value = 5;
  band.frequency.setValueAtTime(from, at);
  band.frequency.linearRampToValueAtTime(to, at + decay);
  const gain = ctx.createGain();
  gain.gain.value = level;

  source.connect(band).connect(gain).connect(master);
  source.start(at);
}

// A piece landing. `snap` is how far the contact dominates the body - above 1 the click leads,
// which is what keeps it from sounding like a struck block.
function strike(ctx, at, { pitch, level, decay, centre, snap = 1.7 }) {
  transient(ctx, at, { level: level * snap, centre: vary(centre, 0.10), decay: 0.005 });
  const f0 = vary(pitch, 0.035);
  MODES.forEach((ratio, i) => {
    mode(ctx, at, f0 * ratio, level * MODE_LEVEL[i], decay * MODE_DECAY[i]);
  });
}

function play(kind) {
  if (muted) return;
  const ctx = context();
  if (!ctx) return;
  const at = ctx.currentTime + 0.005;

  // The four are deliberately different in kind, not one click at four pitches:
  //   move    a dry wooden knock
  //   capture the same knock buried in grit
  //   check   two bright ticks, no body at all - an alert, not a landing
  //   castle  a slide, then a knock
  if (kind === "capture") {
    // Tuned by ear on the bench. A capture turns out not to want grit at all - the crunch is
    // down at 0.01, near enough off - it wants a brighter contact than a move (3600 against
    // 2900), a lower body, and real weight underneath. Brighter and heavier at once, which is
    // not what I would have guessed from the physics.
    crunch(ctx, at, { level: 0.01, centre: 1100, decay: 0.125 });
    strike(ctx, at, { pitch: 390, level: 0.16, decay: 0.060, centre: 3600, snap: 1.8 });
    mode(ctx, at, 180, 0.085, 0.040);            // the weight, which is what carries it
  } else if (kind === "castle") {
    sweep(ctx, at, { level: 0.10, from: 500, to: 1700, decay: 0.085 });
    strike(ctx, at + 0.095, { pitch: 520, level: 0.15, decay: 0.050, centre: 2800 });
  } else if (kind === "check") {
    // no body: two short bright ticks read as a warning rather than as a piece landing
    transient(ctx, at, { level: 0.30, centre: 4200, decay: 0.004 });
    transient(ctx, at + 0.075, { level: 0.26, centre: 4600, decay: 0.004 });
    mode(ctx, at + 0.075, 2100, 0.030, 0.070);
  } else if (kind === "end") {
    mode(ctx, at, 466, 0.09, 0.30);
    mode(ctx, at + 0.15, 311, 0.10, 0.55);
  } else {
    strike(ctx, at, { pitch: 540, level: 0.16, decay: 0.055, centre: 2900, snap: 1.8 });
  }
}

// Which sound a move in algebraic notation wants. The server already sends SAN, so the page
// does not have to work out captures and checks for itself.
function soundForSan(san) {
  if (!san) return "move";
  if (san.includes("#")) return "end";
  if (san.includes("+")) return "check";
  if (san.includes("x")) return "capture";
  if (san.startsWith("O-O")) return "castle";
  return "move";
}

function soundMuted() {
  return muted;
}

function toggleSound() {
  muted = !muted;
  try {
    localStorage.setItem("muted", muted ? "1" : "0");
  } catch { /* nothing to remember it with; the session still honours the toggle */ }
  if (!muted) play("move");   // confirm it is back, the way a volume control does
  return muted;
}
