"""Packs everything the dashboard needs into one JSON file.

The dashboard re-runs the trained model in the browser (trees exported as
plain arrays), so it ships the *inputs* - terrain, per-scene index states,
VIIRS detections, parsed reports - rather than pre-rendered risk maps.
That is what makes the "paste a bulletin, watch the map change" demo
possible, and `tests/parity.mjs` checks the browser numbers match Python.
"""
import base64
import io
import json
import zlib

import numpy as np
from PIL import Image
from scipy import ndimage

from .bulletins import RANGE_HI, season_date
from .config import CFG, COVER_NAMES
from .features import ALL, TEXT, _disk
from .textparse import parse


def b64z(arr):
    return base64.b64encode(zlib.compress(np.ascontiguousarray(arr).tobytes(), 9)).decode()


def png_b64(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def q16(a):
    """Index values in [-1, 1] -> int16, NaN -> -32768."""
    out = np.where(np.isnan(a), -32768, np.round(a * 10000)).astype(np.int16)
    return out


def _finite(x):
    x = float(x)                       # JSON has no Infinity
    return x if np.isfinite(x) else float(np.sign(x) * 1e308)


def export_trees(model):
    trees = []
    for (pred,) in model.clf._predictors:
        nd = pred.nodes
        trees.append([[int(n["feature_idx"]), _finite(n["num_threshold"]),
                       int(n["missing_go_to_left"]), int(n["left"]),
                       int(n["right"]), int(n["is_leaf"]), float(n["value"])]
                      for n in nd])
    return dict(features=model.features,
                feature_index=[ALL.index(f) for f in model.features],
                baseline=float(np.ravel(model.clf._baseline_prediction)[0]),
                neg_rate=model.neg_rate, trees=trees)


def hillshade(elev, cell):
    dy, dx = np.gradient(elev, cell)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(315), np.radians(40)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def scene_states(stack):
    """Carried-forward cloud-free state after each scene."""
    n = stack.ndvi.shape[1]
    cur = {k: np.full((n, n), np.nan, np.float32) for k in ("ndvi", "ndmi", "nbr")}
    last = np.full((n, n), 255, np.uint8)
    states = []
    for i, day in enumerate(stack.days):
        ok = ~np.isnan(stack.ndvi[i])
        for k in cur:
            cur[k][ok] = getattr(stack, k)[i][ok]
        last[ok] = day
        states.append(dict(day=int(day), ndvi=b64z(q16(cur["ndvi"])),
                           ndmi=b64z(q16(cur["ndmi"])), nbr=b64z(q16(cur["nbr"])),
                           last=b64z(last), cloud=float(stack.cloud[i].mean()),
                           rgb=png_b64(stack.rgb[i])))
    return states


def viirs_detections(season, cfg=CFG):
    """The exact detection draws `day_features` uses, as sparse cell lists."""
    out = []
    for d in range(cfg.n_days):
        rng = np.random.default_rng([cfg.seed, season.year, 5, d])
        det = np.zeros((cfg.grid, cfg.grid))
        for back in (1, 2):
            if d - back >= 0:
                det += season.burning[d - back] & (rng.random(det.shape) < 0.7)
        cells = []
        for v in (1, 2):
            cells += np.flatnonzero(det.reshape(-1) == v).tolist() * v
        out.append(sorted(cells))
    return out


def export_dashboard(path, terrain, season, model, attr_baseline, metrics, cfg=CFG):
    D, R = cfg.n_days, len(cfg.ranges)
    shade = hillshade(terrain.elev, cfg.cell_m)

    docs = {}
    for doc in season.docs:
        ex = parse(doc["text"])
        key = f'{doc["day"]}:{doc["range"]}:{doc["kind"]}'
        docs[key] = dict(text=doc["text"], lang=doc["lang"],
                         spans=[[a, b, f] for a, b, f in ex.spans])

    text = [[[None if np.isnan(v) else float(v) for v in season.text[d, r]]
             for r in range(R)] for d in range(D)]

    data = dict(
        meta=dict(grid=cfg.grid, cell_m=cfg.cell_m, bbox=cfg.bbox,
                  season=season.year, n_days=D,
                  dates=[season_date(season.year, d).isoformat() for d in range(D)],
                  ranges=[dict(name=n, hi=RANGE_HI[n], row=r, col=c)
                          for n, r, c in cfg.ranges],
                  covers=COVER_NAMES, text_fields=TEXT, features=ALL),
        terrain=dict(
            elev=b64z(terrain.elev.astype(np.float32)),
            slope=b64z(terrain.slope.astype(np.float32)),
            southness=b64z(terrain.southness.astype(np.float32)),
            cover=b64z(terrain.cover.astype(np.uint8)),
            dist_road=b64z(terrain.dist_road.astype(np.float32)),
            dist_settlement=b64z(terrain.dist_settlement.astype(np.float32)),
            range_id=b64z(terrain.range_id.astype(np.uint8)),
            roads=b64z(terrain.roads.astype(np.uint8)),
            shade=b64z((shade * 255).astype(np.uint8)),
        ),
        scenes=scene_states(season.stack),
        viirs=viirs_detections(season, cfg),
        burning=[np.flatnonzero(season.burning[d].reshape(-1)).tolist() for d in range(D)],
        text=text,
        docs=docs,
        model=export_trees(model),
        attr_baseline=[None if np.isnan(v) else float(v) for v in attr_baseline],
        metrics=metrics,
        disk=np.argwhere(_disk(5) > 0).tolist(),
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
    return data
