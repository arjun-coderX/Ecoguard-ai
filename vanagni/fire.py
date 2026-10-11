"""Fuel moisture and fire-season simulation (the ground truth).

A simple daily fuel-moisture bucket per cell, then human-caused ignitions
(roads and villages matter a lot in Uttarakhand - most fires are lit) and
one-day-per-cell spread to neighbours. Burned cells lose their fuel for
the rest of the season.
"""
import numpy as np
from scipy import ndimage

from .config import CFG, WATER, SETTLEMENT, AGRI, SAL, PINE, OAK, SCRUB

FM_INIT = {WATER: 1.0, SETTLEMENT: 0.2, AGRI: 0.5, SAL: 0.55,
           PINE: 0.45, OAK: 0.72, SCRUB: 0.40}
DRY_RATE = {WATER: 0.0, SETTLEMENT: 0.0, AGRI: 1.0, SAL: 0.95,
            PINE: 1.3, OAK: 0.6, SCRUB: 1.4}
IGNITE_BIAS = {WATER: -99, SETTLEMENT: -99, AGRI: -0.9, SAL: 0.35,
               PINE: 1.0, OAK: -1.6, SCRUB: 0.3}


def _lut(table, cover):
    lut = np.zeros(max(table) + 1)
    for k, v in table.items():
        lut[k] = v
    return lut[cover]


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def simulate_season(terrain, wx, season=2026, cfg=CFG):
    rng = np.random.default_rng([cfg.seed, season, 2])
    D, n = cfg.n_days, cfg.grid
    rid = terrain.range_id
    cover = terrain.cover

    fm = _lut(FM_INIT, cover)
    dry_rate = _lut(DRY_RATE, cover) * (1 + 0.25 * terrain.southness)
    bias = _lut(IGNITE_BIAS, cover)
    burnable = ~np.isin(cover, [WATER, SETTLEMENT])
    road_pull = np.exp(-terrain.dist_road / 600.0)
    village_pull = np.exp(-terrain.dist_settlement / 1500.0)
    pine = (cover == PINE).astype(float)

    fm_hist = np.zeros((D, n, n), np.float32)
    burning = np.zeros((D, n, n), bool)
    burned = np.zeros((n, n), bool)
    kernel = np.ones((3, 3)); kernel[1, 1] = 0

    for d in range(D):
        tmax = wx.tmax[d][rid]
        rh = wx.rh_min[d][rid]
        wind = wx.wind[d][rid]
        rain = wx.rain[d][rid]
        spell = np.minimum(wx.dry_spell[d][rid], 40)
        litter = wx.litter[d][rid]

        dry = 0.010 + 0.0024 * (tmax - 26) - 0.00035 * (rh - 30)
        fm = np.clip(fm - np.clip(dry, 0, None) * dry_rate
                     + np.minimum(0.35, rain * 0.025), 0.03, 1.0)
        fm_hist[d] = fm

        z = (-11.9 + 4.6 * (1 - fm) + bias
             + 1.3 * road_pull + 0.5 * village_pull
             + 0.11 * (tmax - 30) - 0.05 * (rh - 30)
             + 0.04 * (wind - 12) + 0.035 * spell
             + 0.55 * litter * pine + 0.012 * terrain.slope
             - 3.0 * (rain > 2))
        ignite = (rng.random((n, n)) < _sigmoid(z)) & burnable & ~burned

        spread = np.zeros((n, n), bool)
        if d > 0 and burning[d - 1].any():
            nb = ndimage.convolve(burning[d - 1].astype(float), kernel,
                                  mode="constant")
            zs = (-6.8 + 3.4 * (1 - fm) + 0.05 * (wind - 12)
                  - 0.035 * (rh - 30) + 0.8 * pine - 2.5 * (rain > 2))
            p_spread = 1 - (1 - _sigmoid(zs)) ** nb
            spread = (rng.random((n, n)) < p_spread) & burnable & ~burned

        burning[d] = ignite | spread
        burned |= burning[d]

    return fm_hist, burning
