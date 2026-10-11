"""Study-area and run settings.

The default area is a 38 x 38 km box over the Doon valley and the
Mussoorie / Chakrata ranges - the chir pine belt that burns almost
every pre-monsoon season. Everything is expressed on a regular grid
so that a real Sentinel-2 scene can be dropped in with the same shape.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    grid: int = 128                 # cells per side
    cell_m: float = 300.0           # ground size of one cell
    season_start_md: tuple = (3, 15)   # 15 March
    n_days: int = 90                # 15 Mar -> 12 Jun
    revisit_days: int = 5           # Sentinel-2 A+B combined revisit
    train_season: int = 2025        # model is fit on this season ...
    test_season: int = 2026         # ... and evaluated blind on this one
    seed: int = 7

    # forest ranges (name, row, col) - roughly where they sit in the box
    ranges: tuple = field(default_factory=lambda: (
        ("Chakrata", 14, 30),
        ("Kalsi", 52, 14),
        ("Mussoorie", 30, 92),
        ("Raipur", 98, 70),
        ("Narendranagar", 84, 116),
    ))

    # bbox for export (lon/lat, approximate)
    bbox: tuple = (77.80, 30.18, 78.20, 30.53)


COVER_NAMES = ["Water", "Settlement", "Agriculture", "Sal forest",
               "Chir pine", "Oak / broadleaf", "Scrub & grass"]
WATER, SETTLEMENT, AGRI, SAL, PINE, OAK, SCRUB = range(7)

CFG = Config()
