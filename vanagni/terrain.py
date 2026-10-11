"""Static layers: elevation, slope/aspect, land cover, roads, settlements.

In a real deployment these come from SRTM/Copernicus DEM, ESA WorldCover
and OpenStreetMap. Here they are generated so the whole pipeline can run
offline in a few seconds, but the shapes and units match the real ones.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.graph import route_through_array

from .config import CFG, WATER, SETTLEMENT, AGRI, SAL, PINE, OAK, SCRUB


def fractal_noise(n, rng, base=3, octaves=5, persistence=0.55):
    out = np.zeros((n, n))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        k = base * 2 ** o + 1
        coarse = rng.standard_normal((k, k))
        fine = ndimage.zoom(coarse, n / k, order=3)
        out += amp * fine[:n, :n]
        total += amp
        amp *= persistence
    out /= total
    return (out - out.min()) / (out.max() - out.min())


@dataclass
class Terrain:
    elev: np.ndarray          # m
    slope: np.ndarray         # degrees
    aspect: np.ndarray        # degrees clockwise from north (downslope)
    southness: np.ndarray     # 1 = south-facing, -1 = north-facing
    cover: np.ndarray         # int codes, see config.COVER_NAMES
    roads: np.ndarray         # bool
    dist_road: np.ndarray     # m
    dist_settlement: np.ndarray
    range_id: np.ndarray      # which forest range a cell belongs to
    range_elev: np.ndarray    # mean elevation per range (station proxy)


def _slope_aspect(elev, cell):
    dz_drow, dz_dcol = np.gradient(elev, cell)       # rows run north -> south
    slope = np.degrees(np.arctan(np.hypot(dz_dcol, dz_drow)))
    # downslope direction in (east, north) components
    east, north = -dz_dcol, dz_drow
    aspect = (np.degrees(np.arctan2(east, north)) + 360) % 360
    return slope, aspect


def build_terrain(cfg=CFG):
    rng = np.random.default_rng(cfg.seed)
    n = cfg.grid
    rows, cols = np.mgrid[0:n, 0:n] / (n - 1)

    # foothills rise to the north; ridged noise gives spurs and valleys
    ridges = 1 - np.abs(2 * fractal_noise(n, rng, base=3) - 1)
    rough = fractal_noise(n, rng, base=4)
    elev = 380 + 2350 * (0.50 * (1 - rows) ** 1.3 + 0.32 * ridges + 0.18 * rough)

    # a river (think Song / Yamuna tributary) meandering west -> east
    river_row = 0.72 + 0.06 * np.sin(cols * 7.0) + 0.03 * np.sin(cols * 19.0)
    dist_river = np.abs(rows - river_row)
    elev -= 550 * np.exp(-(dist_river / 0.07) ** 2)
    elev = ndimage.gaussian_filter(elev, 1.0)
    elev = np.clip(elev, 350, None)

    slope, aspect = _slope_aspect(elev, cfg.cell_m)
    southness = -np.cos(np.radians(aspect))

    # --- land cover -------------------------------------------------------
    cover = np.full((n, n), SCRUB, dtype=np.int8)
    jitter = (fractal_noise(n, rng, base=6) - 0.5) * 260
    e = elev + jitter
    cover[(e < 900)] = SAL
    pine_band = (e >= 800) & (e < 1950)
    cover[pine_band & (southness > -0.35)] = PINE
    cover[pine_band & (southness <= -0.35)] = OAK
    cover[(e >= 1950) & (e < 2250)] = OAK
    cover[(e >= 2250)] = SCRUB
    flat_low = (elev < 850) & (slope < 12)
    cover[flat_low & (fractal_noise(n, rng, base=5) > 0.35)] = AGRI
    cover[dist_river < 0.012] = WATER

    # settlements: low, flat, near the river
    candidates = np.argwhere((elev < 900) & (slope < 10) & (cover != WATER))
    picks = candidates[rng.choice(len(candidates), 9, replace=False)]
    # plus two hill towns (Mussoorie- and Chakrata-like)
    hill = np.argwhere((elev > 1700) & (elev < 2300) & (slope < 22))
    picks = np.vstack([picks, hill[rng.choice(len(hill), 2, replace=False)]])
    settle = np.zeros((n, n), bool)
    for r, c in picks:
        rad = rng.integers(2, 5)
        rr, cc = np.ogrid[:n, :n]
        settle |= (rr - r) ** 2 + (cc - c) ** 2 <= rad ** 2
    cover[settle & (cover != WATER)] = SETTLEMENT

    # roads: least-cost paths between settlements, steep ground is expensive
    cost = 1 + (slope / 6) ** 2 + (cover == WATER) * 40
    roads = np.zeros((n, n), bool)
    order = picks[np.argsort(picks[:, 1])]
    pairs = list(zip(order[:-1], order[1:])) + [(order[0], order[-1])]
    for a, b in pairs:
        path, _ = route_through_array(cost, tuple(a), tuple(b),
                                      fully_connected=True, geometric=True)
        for r, c in path:
            roads[r, c] = True

    dist_road = ndimage.distance_transform_edt(~roads) * cfg.cell_m
    dist_settlement = ndimage.distance_transform_edt(cover != SETTLEMENT) * cfg.cell_m

    # forest ranges = nearest range HQ
    seeds = np.array([(r, c) for _, r, c in cfg.ranges])
    rr, cc = np.mgrid[0:n, 0:n]
    d = (rr[None] - seeds[:, 0, None, None]) ** 2 + (cc[None] - seeds[:, 1, None, None]) ** 2
    range_id = d.argmin(0).astype(np.int8)
    range_elev = np.array([elev[range_id == k].mean() for k in range(len(seeds))])

    return Terrain(elev, slope, aspect, southness, cover, roads,
                   dist_road, dist_settlement, range_id, range_elev)
