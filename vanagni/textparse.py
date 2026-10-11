"""Turns bulletins and field notes into numbers.

Rule-based on purpose: the vocabulary of IMD bulletins is small and stable,
rules are auditable by a forest officer, they run on a laptop with no GPU,
and every extracted value keeps a character span back to the source text.
(An LLM extractor is a drop-in replacement for `parse` - it only has to
return the same dict.)

Works sentence by sentence: decide what a sentence is *about* from cue
words, then pull the number with the right unit out of it.
"""
import re

import numpy as np

NUM = r"(\d{1,3}(?:\.\d+)?)"
TEMP = re.compile(NUM + r"\s*(?:°\s*C|deg(?:rees)?\s*C|C\b)", re.I)
PCT_RANGE = re.compile(NUM + r"\s*[-–to]+\s*" + NUM + r"\s*(?:%|per\s*cent)", re.I)
PCT = re.compile(NUM + r"\s*(?:%|per\s*cent)", re.I)
SPEED = re.compile(NUM + r"\s*(?:km\s*/\s*h(?:r|our)?|kmph|kmh|किमी)", re.I)
MM = re.compile(NUM + r"\s*(?:mm|मिमी)", re.I)
SENT = re.compile(r"(?:[^.।\n]|(?<=\d)\.(?=\d))+")

WORD_NUM = {w: i for i, w in enumerate(
    ["no", "one", "two", "three", "four", "five", "six", "seven", "eight",
     "nine", "ten"])}

# longest phrases first so "very high" wins over "high"
DANGER = [("very high", 3), ("अति उच्च", 3), ("extreme", 4), ("अत्यधिक", 4),
          ("moderate", 1), ("मध्यम", 1), ("high", 2), ("उच्च", 2),
          ("low", 0), ("निम्न", 0)]
LITTER = [("heavy", 2), ("thick", 2), ("moderate", 1), ("picking up", 1),
          ("light", 0), ("low fuel", 0)]

FORECAST_CUE = re.compile(r"tomorrow|expected|likely|forecast|hover|कल|संभावना", re.I)
TODAY_CUE = re.compile(r"today|recorded|observed|\bwas\b|आज", re.I)
FIELDS = ["obs_tmax", "fc_tmax", "fc_rh", "fc_wind", "fc_gust", "fc_rain",
          "rain_mm", "dry_spell", "danger", "litter", "incidents"]


class Extraction(dict):
    """dict of field -> value, plus .spans = [(start, end, field)]."""
    def __init__(self):
        super().__init__({f: np.nan for f in FIELDS})
        self.spans = []

    def put(self, field, value, start, end):
        if np.isnan(self[field]):
            self[field] = float(value)
            self.spans.append((start, end, field))


def _first(rx, sent, offset):
    m = rx.search(sent)
    if m:
        return float(m.group(1)), offset + m.start(), offset + m.end()
    return None


def _phrase(table, sent, offset):
    low = sent.lower()
    for phrase, val in table:
        i = low.find(phrase)
        if i >= 0:
            return val, offset + i, offset + i + len(phrase)
    return None


def parse(text):
    out = Extraction()
    for m in SENT.finditer(text):
        s, off = m.group(0), m.start()
        low = s.lower()

        # rain that already fell -> dry spell is zero
        if re.search(r"rainfall recorded|mm rain|वर्षा दर्ज", low):
            hit = _first(MM, s, off)
            if hit:
                out.put("rain_mm", *hit)
                out.put("dry_spell", 0, hit[1], hit[2])
            continue
        spell = re.search(r"(?:last|enters day|पिछले)\s*(\d+)|(\d+)\s*consecutive dry", s)
        if spell and re.search(r"dry|rainfall|वर्षा नहीं", low):
            g = 1 if spell.group(1) else 2
            out.put("dry_spell", spell.group(g), off + spell.start(g), off + spell.end(g))
            continue

        if re.search(r"fire danger|fire alert|danger rating|खतरा", low):
            hit = _phrase(DANGER, s, off)
            if hit:
                out.put("danger", *hit)
            continue

        if re.search(r"humidity|\brh\b|आर्द्रता", low):
            rng_m = PCT_RANGE.search(s)
            if rng_m:      # "47-83%" -> afternoon minimum is the lower bound
                lo = min(float(rng_m.group(1)), float(rng_m.group(2)))
                out.put("fc_rh", lo, off + rng_m.start(), off + rng_m.end())
            else:
                hit = _first(PCT, s, off)
                if hit:
                    out.put("fc_rh", *hit)
            continue

        if re.search(r"wind|हवा", low):
            speeds = list(SPEED.finditer(s))
            if speeds:
                a = speeds[0]
                out.put("fc_wind", a.group(1), off + a.start(), off + a.end())
            if re.search(r"gust|झोंके", low) and len(speeds) > 1:
                b = speeds[1]
                out.put("fc_gust", b.group(1), off + b.start(), off + b.end())
            continue

        t = TEMP.search(s)
        if t and re.search(r"temp|tmax|तापमान", low):
            field = "fc_tmax" if FORECAST_CUE.search(s) else "obs_tmax"
            out.put(field, t.group(1), off + t.start(), off + t.end())
            continue

        dry_fc = re.search(r"no rain expected|dry weather|clear sky|शुष्क रहेगा", low)
        wet_fc = re.search(r"(rain|shower|thunder|बौछारें).*(likely|possible|chance|संभव)|chance of", low)
        if dry_fc:
            out.put("fc_rain", 0, off + dry_fc.start(), off + dry_fc.end())
            continue
        if wet_fc:
            out.put("fc_rain", 1, off + wet_fc.start(), off + wet_fc.end())
            continue

        # field-note sentences
        if re.search(r"litter|needle|fuel load", low):
            hit = _phrase(LITTER, s, off)
            if hit:
                out.put("litter", *hit)
            continue
        if re.search(r"fire", low):
            if re.search(r"\bnil\b|\bno fire", low):
                k = re.search(r"nil|no fire", low)
                out.put("incidents", 0, off + k.start(), off + k.end())
                continue
            k = re.search(r"\b(\d+|" + "|".join(WORD_NUM) + r")\b", low)
            if k:
                tok = k.group(1)
                val = int(tok) if tok.isdigit() else WORD_NUM[tok]
                out.put("incidents", val, off + k.start(), off + k.end())
    return out


def score_extraction(docs):
    """Field-level accuracy of the parser against the generator's truth."""
    keymap = {"fc_tmax": "fc_tmax", "fc_rh": "fc_rh", "fc_wind": "fc_wind",
              "fc_rain": "fc_rain", "dry_spell": "dry_spell", "danger": "danger",
              "litter": "litter", "incidents": "incidents"}
    hits, total, missing = {}, {}, {}
    for doc in docs:
        ex = parse(doc["text"])
        for f, tk in keymap.items():
            if tk not in doc["truth"]:
                continue
            total[f] = total.get(f, 0) + 1
            if np.isnan(ex[f]):
                missing[f] = missing.get(f, 0) + 1
            elif abs(ex[f] - doc["truth"][tk]) < 0.51:
                hits[f] = hits.get(f, 0) + 1
    rows = {}
    for f in total:
        present = total[f] - missing.get(f, 0)
        rows[f] = dict(found=present / total[f],
                       correct_when_found=hits.get(f, 0) / max(present, 1))
    return rows
