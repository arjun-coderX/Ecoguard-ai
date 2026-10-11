// Checks the browser engine against Python: same parsed report values,
// same risk predictions. Run after scripts/run_pipeline.py:
//   node tests/parity.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
await import(join(root, "dashboard", "engine.js"));
const V = globalThis.Vanagni;

function readNpy(path) {
  const buf = readFileSync(path);
  const headerLen = buf.readUInt16LE(8);
  const header = buf.subarray(10, 10 + headerLen).toString();
  const body = buf.subarray(10 + headerLen);
  const ab = body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength);
  return header.includes("<f4") ? new Float32Array(ab) : new BigInt64Array(ab);
}

const data = JSON.parse(readFileSync(join(root, "outputs", "dashboard_data.json"), "utf8"));

// 1. parser parity on every bulletin of the test season
const TF = data.meta.text_fields;
const bulletinFields = ["fc_tmax", "fc_rh", "fc_wind", "fc_gust", "fc_rain", "dry_spell", "danger", "obs_tmax"];
let checked = 0, bad = 0;
for (const [key, doc] of Object.entries(data.docs)) {
  const [day, range, kind] = key.split(":");
  if (kind !== "bulletin") continue;
  const js = V.parse(doc.text).values;
  for (const f of bulletinFields) {
    const py = data.text[+day][+range][TF.indexOf(f)];
    const a = py === null ? NaN : py, b = js[f];
    checked++;
    if (!(Number.isNaN(a) && Number.isNaN(b)) && Math.abs(a - b) > 1e-4) {
      bad++;
      if (bad < 5) console.log("parser mismatch", key, f, a, b);
    }
  }
}
console.log(`parser: ${checked - bad}/${checked} field values identical (float32 tolerance)`);

// 2. model parity on the saved peak day
const eng = await V.load(data);
const day = Number(readNpy(join(root, "outputs", "parity_day.npy"))[0]);
const py = readNpy(join(root, "outputs", "parity_pred.npy"));
const t0 = performance.now();
const { risk } = V.run(eng, day);
const ms = performance.now() - t0;
let maxAbs = 0;
for (let i = 0; i < py.length; i++) maxAbs = Math.max(maxAbs, Math.abs(py[i] - risk[i]));
console.log(`model: day ${day}, max |python - js| = ${maxAbs.toExponential(2)}, run+attribution ${ms.toFixed(0)} ms`);
if (bad > 0 || maxAbs > 1e-5) process.exit(1);
console.log("parity OK");
