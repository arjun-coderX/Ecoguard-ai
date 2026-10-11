"""Builds the cell x day feature table that fuses imagery with text.

For a target day d every feature is something that was actually available
the evening before:

  terrain   static layers (DEM, land cover, roads, villages)
  imagery   latest cloud-free Sentinel-2 indices up to d-1, their change
            over ~2 weeks, and VIIRS active-fire detections nearby on d-1/d-2
  text      the range's weather bulletin issued on d-1 (tomorrow's forecast)
            and its most recent weekly field note

The label is whether the cell is burning on day d.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .bulletins import write_reports
from .config import CFG, WATER, SETTLEMENT
from .fire import simulate_season
from .imagery import synthesise_scenes, compute_indices, latest_clear
from .textparse import parse
from .weather import simulate_weather

TERRAIN = ["elev", "slope", "southness", "cover", "dist_road", "dist_settlement"]
IMAGERY = ["ndvi", "ndmi", "nbr", "d_ndmi_15", "img_age", "viirs_nearby"]
TEXT = ["fc_tmax", "fc_rh", "fc_wind", "fc_gust", "fc_rain", "dry_spell",
        "danger", "obs_tmax", "litter", "incidents"]
ALL = TERRAIN + IMAGERY + TEXT
CATEGORICAL = []          # cover codes split fine as ordinal; keeps trees exportable

BULLETIN_FIELDS = ["fc_tmax", "fc_rh", "fc_wind", "fc_gust", "fc_rain",
                   "dry_spell", "danger", "obs_tmax"]
NOTE_FIELDS = ["litter", "incidents"]


@dataclass
class Season:
    year: int
    weather: object
    fm: np.ndarray
    burning: np.ndarray
    stack: object
    docs: list
    text: np.ndarray            # (days, ranges, len(TEXT)) parsed values
    parsed: dict                # (day, range, kind) -> Extraction


def _disk(radius):
    r = int(radius)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2 <= radius ** 2).astype(float)


def simulate(terrain, year, cfg=CFG):
    wx = simulate_weather(terrain, year, cfg)
    fm, burning = simulate_season(terrain, wx, year, cfg)
    days, scenes = synthesise_scenes(terrain, fm, burning, year, cfg)
    stack = compute_indices(days, scenes)
    docs = write_reports(terrain, wx, burning, year, cfg)

    D, R = cfg.n_days, len(cfg.ranges)
    text = np.full((D, R, len(TEXT)), np.nan, np.float32)
    parsed = {}
    for doc in docs:
        ex = parse(doc["text"])
        parsed[(doc["day"], doc["range"], doc["kind"])] = ex
        fields = BULLETIN_FIELDS if doc["kind"] == "bulletin" else NOTE_FIELDS
        for f in fields:
            text[doc["day"], doc["range"], TEXT.index(f)] = ex[f]
    # field notes are weekly - carry the latest one forward
    for f in NOTE_FIELDS:
        j = TEXT.index(f)
        for r in range(R):
            last = np.nan
            for d in range(D):
                if not np.isnan(text[d, r, j]):
                    last = text[d, r, j]
                text[d, r, j] = last
    return Season(year, wx, fm, burning, stack, docs, text, parsed)


def day_features(terrain, season, d, cfg=CFG):
    """Feature matrix (cells, len(ALL)) for target day d (uses info <= d-1)."""
    rng = np.random.default_rng([cfg.seed, season.year, 5, d])
    n2 = cfg.grid ** 2
    cols = {}
    cols["elev"] = terrain.elev
    cols["slope"] = terrain.slope
    cols["southness"] = terrain.southness
    cols["cover"] = terrain.cover.astype(float)
    cols["dist_road"] = terrain.dist_road
    cols["dist_settlement"] = terrain.dist_settlement

    st = season.stack
    ndvi, age = latest_clear(st, "ndvi", d - 1)
    ndmi, _ = latest_clear(st, "ndmi", d - 1)
    nbr, _ = latest_clear(st, "nbr", d - 1)
    ndmi_old, _ = latest_clear(st, "ndmi", d - 16)
    cols["ndvi"], cols["ndmi"], cols["nbr"] = ndvi, ndmi, nbr
    cols["d_ndmi_15"] = ndmi - ndmi_old
    cols["img_age"] = age.astype(float)

    # VIIRS 375 m active fire: sees ~70% of burning cells, no clouds modelled
    det = np.zeros((cfg.grid, cfg.grid))
    for back in (1, 2):
        if d - back >= 0:
            det += season.burning[d - back] & (rng.random(det.shape) < 0.7)
    cols["viirs_nearby"] = ndimage.convolve(det, _disk(5), mode="constant")

    prev = max(d - 1, 0)
    per_range = season.text[prev]                       # (R, len(TEXT))
    cell_text = per_range[terrain.range_id]             # (H, W, len(TEXT))
    for j, f in enumerate(TEXT):
        cols[f] = cell_text[..., j]

    X = np.stack([np.asarray(cols[f], np.float32).reshape(n2) for f in ALL], axis=1)
    y = season.burning[d].reshape(n2)
    return X, y


def burnable_mask(terrain):
    return ~np.isin(terrain.cover, [WATER, SETTLEMENT]).reshape(-1)


def season_table(terrain, season, neg_rate=None, cfg=CFG):
    """Stack days 1..D-1. Optionally subsample negatives (training)."""
    rng = np.random.default_rng([cfg.seed, season.year, 6])
    keep_cells = burnable_mask(terrain)
    Xs, ys, ds, cs = [], [], [], []
    cell_idx = np.arange(cfg.grid ** 2)
    for d in range(1, cfg.n_days):
        X, y = day_features(terrain, season, d, cfg)
        m = keep_cells.copy()
        if neg_rate is not None:
            m &= y | (rng.random(len(y)) < neg_rate)
        Xs.append(X[m]); ys.append(y[m])
        ds.append(np.full(m.sum(), d, np.int16)); cs.append(cell_idx[m])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ds), np.concatenate(cs)
