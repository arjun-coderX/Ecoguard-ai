"""Daily station weather for each forest range.

This is the *ground truth* the bulletins are written from. The model never
sees these arrays directly - it only sees what survives the trip through a
text bulletin and the parser, which is the point of the exercise.
"""
from dataclasses import dataclass

import numpy as np

from .config import CFG

LAPSE = 6.5 / 1000          # degC per metre


@dataclass
class Weather:
    tmax: np.ndarray        # (days, ranges)
    tmax_normal: np.ndarray
    rh_min: np.ndarray
    rh_max: np.ndarray
    wind: np.ndarray        # km/h, mean afternoon
    gust: np.ndarray        # km/h
    wind_dir: np.ndarray    # compass string
    rain: np.ndarray        # mm in 24h
    dry_spell: np.ndarray   # consecutive days with < 1 mm
    litter: np.ndarray      # 0 light, 1 moderate, 2 heavy (pine needle fall)


DIRS = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])


def simulate_weather(terrain, season=2026, cfg=CFG):
    rng = np.random.default_rng([cfg.seed, season, 1])
    D, R = cfg.n_days, len(cfg.ranges)
    t = np.arange(D)

    # seasonal normal at 600 m: ~29 C mid-March to ~37 C early June
    normal_600 = 29 + 8.5 * (1 - np.exp(-t / 45)) / (1 - np.exp(-D / 45))
    tmax_normal = normal_600[:, None] - LAPSE * (terrain.range_elev[None] - 600)

    # synoptic anomaly shared by all ranges (AR(1)) + two heat spells
    anom = np.zeros(D)
    for d in range(1, D):
        anom[d] = 0.75 * anom[d - 1] + rng.normal(0, 1.1)
    spells = [(rng.integers(30, 46), 5, rng.uniform(2.5, 4.0)),
              (rng.integers(58, 72), 6, rng.uniform(3.5, 5.0))]
    for centre, width, amp in spells:
        anom += amp * np.exp(-((t - centre) / width) ** 2)

    # rain: western disturbances early, pre-monsoon thunderstorms late
    rain = np.zeros((D, R))
    wd_days = sorted(rng.choice(np.arange(0, 55), 6, replace=False))
    storm_days = sorted(rng.choice(np.arange(68, D), 6, replace=False))
    for d in wd_days + storm_days:
        wet = rng.random(R) < (0.9 if d in wd_days else 0.55)
        rain[d] = wet * rng.gamma(1.6, 7.0 if d in wd_days else 9.0, R)
    rain[rain < 1] = 0
    # wet days are cooler
    cooling = -2.5 * (rain > 0)

    tmax = (tmax_normal + anom[:, None] + cooling
            + rng.normal(0, 0.7, (D, R)))
    tmax = np.round(tmax, 1)

    # humidity falls as it heats up, recovers after rain
    rh_min = 48 - 1.3 * (tmax - tmax_normal) - 0.22 * t[:, None] + 22 * (rain > 0)
    rh_min += rng.normal(0, 4, (D, R))
    rh_min = np.clip(np.round(rh_min), 9, 85).astype(int)
    rh_max = np.clip(rh_min + rng.integers(18, 34, (D, R)), 30, 98)

    base_wind = 9 + 6 * (t / D)[:, None] + rng.gamma(2.0, 2.4, (D, R))
    # storm days are gusty
    base_wind += 8 * np.isin(t, storm_days)[:, None]
    wind = np.round(base_wind).astype(int)
    gust = wind + rng.integers(6, 20, (D, R)) * (rng.random((D, R)) < 0.35)

    # pre-monsoon winds are mostly westerly / north-westerly
    dir_idx = rng.choice(8, (D, R), p=[.06, .03, .03, .04, .06, .16, .30, .32])
    wind_dir = DIRS[dir_idx]

    dry_spell = np.zeros((D, R), int)
    start = rng.integers(4, 12, R)   # days dry before the season window
    for d in range(D):
        prev = start if d == 0 else dry_spell[d - 1]
        dry_spell[d] = np.where(rain[d] >= 1, 0, prev + 1)

    # chir pine needle fall builds through April-May
    litter_score = (t / D)[:, None] * 2.4 + rng.normal(0, 0.35, (D, R))
    litter = np.clip(np.round(litter_score), 0, 2).astype(int)

    return Weather(tmax, tmax_normal, rh_min, rh_max, wind, gust, wind_dir,
                   rain, dry_spell, litter)
