// Vanagni browser engine: report parser, feature builder, tree inference.
// Mirrors vanagni/textparse.py, features.py and the exported
// HistGradientBoosting model. tests/parity.mjs checks it against Python.
(function (root) {
  "use strict";

  // ---------- decoding --------------------------------------------------
  function b64ToBytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  async function inflate(b64) {
    const ds = new DecompressionStream("deflate");
    const stream = new Blob([b64ToBytes(b64)]).stream().pipeThrough(ds);
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  async function f32(b64) { return new Float32Array((await inflate(b64)).buffer); }
  async function u8(b64) { return inflate(b64); }
  async function i16idx(b64) {
    const raw = new Int16Array((await inflate(b64)).buffer);
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = raw[i] === -32768 ? NaN : raw[i] / 10000;
    return out;
  }

  // ---------- parser (port of textparse.py) ------------------------------
  const NUM = String.raw`(\d{1,3}(?:\.\d+)?)`;
  const TEMP = new RegExp(NUM + String.raw`\s*(?:°\s*C|deg(?:rees)?\s*C|C\b)`, "i");
  const PCT_RANGE = new RegExp(NUM + String.raw`\s*[-–to]+\s*` + NUM + String.raw`\s*(?:%|per\s*cent)`, "i");
  const PCT = new RegExp(NUM + String.raw`\s*(?:%|per\s*cent)`, "i");
  const SPEED_G = new RegExp(NUM + String.raw`\s*(?:km\s*/\s*h(?:r|our)?|kmph|kmh|किमी)`, "gi");
  const MM = new RegExp(NUM + String.raw`\s*(?:mm|मिमी)`, "i");
  const SENT = /(?:[^.।\n]|(?<=\d)\.(?=\d))+/g;
  const WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
  const DANGER = [["very high", 3], ["अति उच्च", 3], ["extreme", 4], ["अत्यधिक", 4],
    ["moderate", 1], ["मध्यम", 1], ["high", 2], ["उच्च", 2], ["low", 0], ["निम्न", 0]];
  const LITTER = [["heavy", 2], ["thick", 2], ["moderate", 1], ["picking up", 1],
    ["light", 0], ["low fuel", 0]];
  const FORECAST_CUE = /tomorrow|expected|likely|forecast|hover|कल|संभावना/i;
  const FIELDS = ["obs_tmax", "fc_tmax", "fc_rh", "fc_wind", "fc_gust", "fc_rain",
    "rain_mm", "dry_spell", "danger", "litter", "incidents"];

  function parse(text) {
    const out = {}; FIELDS.forEach(f => (out[f] = NaN));
    const spans = [];
    const put = (f, v, a, b) => {
      if (Number.isNaN(out[f])) { out[f] = Number(v); spans.push([a, b, f]); }
    };
    const first = (rx, s, off) => {
      const m = rx.exec(s);
      return m ? [Number(m[1]), off + m.index, off + m.index + m[0].length] : null;
    };
    const phrase = (table, s, off) => {
      const low = s.toLowerCase();
      for (const [p, v] of table) {
        const i = low.indexOf(p);
        if (i >= 0) return [v, off + i, off + i + p.length];
      }
      return null;
    };
    for (const m of text.matchAll(SENT)) {
      const s = m[0], off = m.index, low = s.toLowerCase();
      let hit;
      if (/rainfall recorded|mm rain|वर्षा दर्ज/.test(low)) {
        hit = first(MM, s, off);
        if (hit) { put("rain_mm", ...hit); put("dry_spell", 0, hit[1], hit[2]); }
        continue;
      }
      const sp = /(?:last|enters day|पिछले)\s*(\d+)|(\d+)\s*consecutive dry/.exec(s);
      if (sp && /dry|rainfall|वर्षा नहीं/.test(low)) {
        const g = sp[1] !== undefined ? 1 : 2;
        const start = off + sp.index + sp[0].indexOf(sp[g]);
        put("dry_spell", sp[g], start, start + sp[g].length);
        continue;
      }
      if (/fire danger|fire alert|danger rating|खतरा/.test(low)) {
        hit = phrase(DANGER, s, off); if (hit) put("danger", ...hit); continue;
      }
      if (/humidity|\brh\b|आर्द्रता/.test(low)) {
        const r = PCT_RANGE.exec(s);
        if (r) put("fc_rh", Math.min(+r[1], +r[2]), off + r.index, off + r.index + r[0].length);
        else { hit = first(PCT, s, off); if (hit) put("fc_rh", ...hit); }
        continue;
      }
      if (/wind|हवा/.test(low)) {
        const speeds = [...s.matchAll(SPEED_G)];
        if (speeds.length) put("fc_wind", speeds[0][1], off + speeds[0].index, off + speeds[0].index + speeds[0][0].length);
        if (/gust|झोंके/.test(low) && speeds.length > 1)
          put("fc_gust", speeds[1][1], off + speeds[1].index, off + speeds[1].index + speeds[1][0].length);
        continue;
      }
      const t = TEMP.exec(s);
      if (t && /temp|tmax|तापमान/.test(low)) {
        put(FORECAST_CUE.test(s) ? "fc_tmax" : "obs_tmax", t[1], off + t.index, off + t.index + t[0].length);
        continue;
      }
      const dry = /no rain expected|dry weather|clear sky|शुष्क रहेगा/.exec(low);
      const wet = /(rain|shower|thunder|बौछारें).*(likely|possible|chance|संभव)|chance of/.exec(low);
      if (dry) { put("fc_rain", 0, off + dry.index, off + dry.index + dry[0].length); continue; }
      if (wet) { put("fc_rain", 1, off + wet.index, off + wet.index + wet[0].length); continue; }
      if (/litter|needle|fuel load/.test(low)) {
        hit = phrase(LITTER, s, off); if (hit) put("litter", ...hit); continue;
      }
      if (/fire/.test(low)) {
        const nil = /\bnil\b|\bno fire/.exec(low);
        if (nil) { put("incidents", 0, off + nil.index, off + nil.index + nil[0].length); continue; }
        const k = new RegExp(String.raw`\b(\d+|` + WORDS.join("|") + String.raw`)\b`).exec(low);
        if (k) put("incidents", /^\d+$/.test(k[1]) ? +k[1] : WORDS.indexOf(k[1]), off + k.index, off + k.index + k[0].length);
      }
    }
    return { values: out, spans };
  }

  // ---------- data model -------------------------------------------------
  async function load(data) {
    const n2 = data.meta.grid * data.meta.grid;
    const T = data.terrain;
    const terrain = {
      elev: await f32(T.elev), slope: await f32(T.slope), southness: await f32(T.southness),
      cover: await u8(T.cover), dist_road: await f32(T.dist_road),
      dist_settlement: await f32(T.dist_settlement), range_id: await u8(T.range_id),
      roads: await u8(T.roads), shade: await u8(T.shade),
    };
    const scenes = [];
    for (const s of data.scenes) {
      scenes.push({ day: s.day, cloud: s.cloud, rgb: s.rgb,
        ndvi: await i16idx(s.ndvi), ndmi: await i16idx(s.ndmi),
        nbr: await i16idx(s.nbr), last: await u8(s.last) });
    }
    return { data, n2, terrain, scenes };
  }

  function sceneAt(scenes, day) {          // last scene with acquisition <= day
    let k = -1;
    for (let i = 0; i < scenes.length; i++) if (scenes[i].day <= day) k = i;
    return k;
  }

  // Build the feature matrix for target day d (row-major cells x features).
  // `textOverride` optionally replaces a range's parsed report values.
  function features(eng, d, textOverride) {
    const { data, n2, terrain, scenes } = eng;
    const F = data.meta.features, nf = F.length, grid = data.meta.grid;
    const X = new Float32Array(n2 * nf);
    const col = Object.fromEntries(F.map((f, j) => [f, j]));
    const a = sceneAt(scenes, d - 1), b = sceneAt(scenes, d - 16);
    const viirs = new Float32Array(n2);
    for (const c of data.viirs[d]) {
      const r0 = Math.floor(c / grid), c0 = c % grid;
      for (const [i, j] of data.disk) {
        const r = r0 + i - 5, cc = c0 + j - 5;
        if (r >= 0 && r < grid && cc >= 0 && cc < grid) viirs[r * grid + cc] += 1;
      }
    }
    const prev = Math.max(d - 1, 0);
    const TF = data.meta.text_fields;
    const textRows = data.text[prev].map((row, r) => {
      const o = textOverride && textOverride[r];
      return TF.map((f, j) => {
        const v = o && o[f] !== undefined ? o[f] : row[j];
        return v === null || v === undefined ? NaN : v;
      });
    });
    for (let i = 0; i < n2; i++) {
      const o = i * nf;
      X[o + col.elev] = terrain.elev[i];
      X[o + col.slope] = terrain.slope[i];
      X[o + col.southness] = terrain.southness[i];
      X[o + col.cover] = terrain.cover[i];
      X[o + col.dist_road] = terrain.dist_road[i];
      X[o + col.dist_settlement] = terrain.dist_settlement[i];
      if (a >= 0) {
        const S = scenes[a];
        X[o + col.ndvi] = S.ndvi[i]; X[o + col.ndmi] = S.ndmi[i]; X[o + col.nbr] = S.nbr[i];
        X[o + col.d_ndmi_15] = b >= 0 ? S.ndmi[i] - scenes[b].ndmi[i] : NaN;
        X[o + col.img_age] = S.last[i] === 255 ? 99 : (d - 1) - S.last[i];
      } else {
        X[o + col.ndvi] = X[o + col.ndmi] = X[o + col.nbr] = X[o + col.d_ndmi_15] = NaN;
        X[o + col.img_age] = 99;
      }
      X[o + col.viirs_nearby] = viirs[i];
      const tr = textRows[terrain.range_id[i]];
      for (let j = 0; j < TF.length; j++) X[o + col[TF[j]]] = tr[j];
    }
    return X;
  }

  function predict(model, X, nf, n) {
    const out = new Float32Array(n);
    const fi = model.feature_index, w = model.neg_rate;
    for (let i = 0; i < n; i++) {
      let raw = model.baseline;
      const o = i * nf;
      for (const tree of model.trees) {
        let k = 0;
        for (;;) {
          const nd = tree[k];
          if (nd[5]) { raw += nd[6]; break; }
          const x = X[o + fi[nd[0]]];
          k = Number.isNaN(x) ? (nd[2] ? nd[3] : nd[4]) : (x <= nd[1] ? nd[3] : nd[4]);
        }
      }
      const p = 1 / (1 + Math.exp(-raw));
      out[i] = (p * w) / (p * w + (1 - p));
    }
    return out;
  }

  const GROUPS = {
    terrain: ["elev", "slope", "southness", "cover", "dist_road", "dist_settlement"],
    satellite: ["ndvi", "ndmi", "nbr", "d_ndmi_15", "img_age", "viirs_nearby"],
    reports: ["fc_tmax", "fc_rh", "fc_wind", "fc_gust", "fc_rain", "dry_spell",
      "danger", "obs_tmax", "litter", "incidents"],
  };

  function risk(eng, d, textOverride) {
    const nf = eng.data.meta.features.length;
    const X = features(eng, d, textOverride);
    return { X, risk: predict(eng.data.model, X, nf, eng.n2) };
  }

  // Occlusion attribution per data source (as in model.py).
  function attribute(eng, X, base) {
    const { data, n2 } = eng, F = data.meta.features, nf = F.length;
    const attr = {};
    for (const [g, feats] of Object.entries(GROUPS)) {
      const Xo = X.slice();
      const idx = feats.map(f => F.indexOf(f));
      for (let i = 0; i < n2; i++)
        for (const j of idx) { const b = data.attr_baseline[j]; Xo[i * nf + j] = b === null ? NaN : b; }
      const p = predict(data.model, Xo, nf, n2);
      const a = new Float32Array(n2);
      for (let i = 0; i < n2; i++) a[i] = base[i] - p[i];
      attr[g] = a;
    }
    return attr;
  }

  function run(eng, d, textOverride) {
    const r = risk(eng, d, textOverride);
    return { ...r, attr: attribute(eng, r.X, r.risk) };
  }

  root.Vanagni = { load, parse, features, predict, risk, attribute, run, GROUPS, sceneAt };
})(typeof window !== "undefined" ? window : globalThis);
