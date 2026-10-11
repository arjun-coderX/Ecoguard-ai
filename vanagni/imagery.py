"""Sentinel-2-like surface reflectance composites and spectral indices.

Bands follow Sentinel-2 L2A naming: B04 red, B08 NIR, B11 SWIR-1, B12 SWIR-2.
Scenes arrive every `revisit_days`, are partly cloudy, and clouds are masked
the usual way (bright in every band, flat spectrum). Downstream code only
ever sees the masked, carried-forward index stack, exactly as it would with
a real scene archive.

To use real data, replace `synthesise_scenes` with a loader that returns the
same (n_scenes, 4, H, W) array and a list of acquisition days - see
`load_geotiff_stack` at the bottom.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import CFG, WATER, SETTLEMENT, AGRI, SAL, PINE, OAK, SCRUB
from .terrain import fractal_noise

#                 B04    B08    B11    B12
BASE = {WATER:    (0.030, 0.020, 0.010, 0.008),
        SETTLEMENT: (0.150, 0.220, 0.280, 0.240),
        AGRI:     (0.060, 0.320, 0.200, 0.120),
        SAL:      (0.040, 0.330, 0.170, 0.080),
        PINE:     (0.035, 0.280, 0.150, 0.080),
        OAK:      (0.030, 0.360, 0.160, 0.070),
        SCRUB:    (0.080, 0.240, 0.250, 0.170)}
HARVESTED = (0.140, 0.220, 0.300, 0.250)   # wheat cut from mid-April
SCAR = (0.060, 0.100, 0.180, 0.230)        # fresh burn scar
BANDS = ["B04", "B08", "B11", "B12"]


@dataclass
class IndexStack:
    days: np.ndarray            # acquisition day of each scene
    ndvi: np.ndarray            # (scenes, H, W), NaN where cloudy
    ndmi: np.ndarray
    nbr: np.ndarray
    cloud: np.ndarray           # bool mask
    rgb: np.ndarray             # (scenes, H, W, 3) false colour, uint8


def synthesise_scenes(terrain, fm_hist, burning, season, cfg=CFG):
    rng = np.random.default_rng([cfg.seed, season, 3])
    n = cfg.grid
    days = np.arange(2, cfg.n_days, cfg.revisit_days)
    burned_by = np.cumsum(burning, axis=0) > 0

    base = np.zeros((4, n, n))
    for k, refl in BASE.items():
        base[:, terrain.cover == k] = np.array(refl)[:, None]
    # within-class texture so it doesn't look like a paint-by-numbers map
    texture = 0.88 + 0.24 * fractal_noise(n, rng, base=12, octaves=3)
    illum = 0.85 + 0.15 * np.cos(np.radians(terrain.slope)) * \
        (0.5 + 0.5 * np.cos(np.radians(terrain.aspect - 135)))

    scenes = np.zeros((len(days), 4, n, n), np.float32)
    for i, d in enumerate(days):
        dry = 1 - fm_hist[d]
        refl = base * texture
        if d >= 32:      # mid-April harvest
            agri = terrain.cover == AGRI
            refl[:, agri] = np.array(HARVESTED)[:, None] * texture[agri]
        veg = np.isin(terrain.cover, [SAL, PINE, OAK, SCRUB, AGRI])
        refl[0, veg] += 0.03 * dry[veg]
        refl[1, veg] *= 0.72 + 0.28 * (1 - dry[veg])
        refl[2, veg] += 0.11 * dry[veg]
        refl[3, veg] += 0.08 * dry[veg]
        scar = burned_by[d]
        refl[:, scar] = np.array(SCAR)[:, None] * texture[scar]
        refl *= illum
        refl += rng.normal(0, 0.006, refl.shape)

        # clouds: more frequent early (WDs) and late (pre-monsoon build-up)
        cover_frac = 0.08 + 0.35 * (d < 20 or d > 75) * rng.random()
        cloud_field = fractal_noise(n, rng, base=4, octaves=4)
        cloud = cloud_field > np.quantile(cloud_field, 1 - cover_frac)
        haze = ndimage.gaussian_filter(cloud.astype(float), 2)
        refl = refl * (1 - haze) + 0.42 * haze
        scenes[i] = np.clip(refl, 0, 1)
    return days, scenes


def cloud_mask(scene):
    red, nir, sw1, sw2 = scene
    bright = (red > 0.25) & (nir > 0.28) & (sw1 > 0.25)
    flat = np.abs(nir - red) / (nir + red + 1e-6) < 0.25
    return bright & flat


def compute_indices(days, scenes):
    eps = 1e-6
    red, nir, sw1, sw2 = (scenes[:, i] for i in range(4))
    ndvi = (nir - red) / (nir + red + eps)
    ndmi = (nir - sw1) / (nir + sw1 + eps)
    nbr = (nir - sw2) / (nir + sw2 + eps)
    cloud = np.stack([cloud_mask(s) for s in scenes])
    for arr in (ndvi, ndmi, nbr):
        # store like an L2A product: scaled int16 (1e-4 steps)
        arr[:] = np.round(arr * 10000) / np.float32(10000)
        arr[cloud] = np.nan
    rgb = false_colour(scenes)
    return IndexStack(days, ndvi, ndmi, nbr, cloud, rgb)


def false_colour(scenes):
    """SWIR-NIR-Red composite: burn scars dark red, healthy forest green."""
    sw1, nir, red = scenes[:, 2], scenes[:, 1], scenes[:, 0]
    rgb = np.stack([sw1 * 2.6, nir * 2.3, red * 4.0], axis=-1)
    rgb = np.clip(rgb, 0, 1) ** 0.85
    return (rgb * 255).astype(np.uint8)


def latest_clear(stack, attr, upto_day):
    """Most recent cloud-free value of an index per pixel, plus its age."""
    arr = getattr(stack, attr)
    use = stack.days <= upto_day
    value = np.full(arr.shape[1:], np.nan, np.float32)
    age = np.full(arr.shape[1:], 99, np.int16)
    for i in np.where(use)[0]:
        ok = ~np.isnan(arr[i])
        value[ok] = arr[i][ok]
        age[ok] = upto_day - stack.days[i]
    return value, age


def load_geotiff_stack(paths, days):
    """Load real Sentinel-2 L2A scenes (one 4-band GeoTIFF per date).

    Each file must hold B04, B08, B11, B12 resampled to a common grid,
    scaled to reflectance (divide L2A DN by 10000). Needs `rasterio`.
    """
    import rasterio  # optional dependency
    out = []
    for p in paths:
        with rasterio.open(p) as src:
            out.append(src.read().astype(np.float32))
    return np.asarray(days), np.stack(out)
