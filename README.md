# EcoGuard AI (Vanagni)

Made by Arjun Ray · arjun.onlywork@gmail.com

*वनाग्नि — forest fire*

Next-day wildfire risk for Uttarakhand's chir pine belt, at 300 m resolution, built
by reading two things together: what the satellite sees on the ground, and what the
weather bulletins and range officers say in writing.

Every pre-monsoon season (March to mid-June) the pine forests around Dehradun,
Mussoorie and Tehri burn. Most fires are lit by people, they run along the dry
needle litter, and a forest division has a handful of crews for hundreds of square
kilometres. The useful question for a Divisional Forest Officer at 6 pm is simple:
**where should my crews be tomorrow?**

## What it does

For every 300 m cell, every evening, Vanagni outputs the probability that the cell
is burning the next day. It uses only what was available that evening:

| Source | What we take from it |
|---|---|
| Sentinel-2 L2A (5-day revisit) | NDVI, NDMI (canopy moisture), NBR (burn scars), 15-day NDMI change, days since last cloud-free look |
| VIIRS 375 m active fire | detections within 1.5 km over the last two nights |
| DEM, land cover, OSM | elevation, slope, aspect, cover class, distance to road and village |
| IMD-style evening bulletin (English **or Hindi**) | tomorrow's max temperature, minimum humidity, wind and gusts, dry-spell length, rain outlook, fire-danger level |
| Weekly range-officer field note | pine-needle litter (light/moderate/heavy), fire incidents attended |

The text is parsed sentence by sentence with rules (`vanagni/textparse.py`). Each
extracted number keeps a character span back to the source sentence, so the
dashboard can highlight exactly what was read. Fields a bulletin leaves out stay
missing; the model handles that natively.

A gradient-boosted tree model fuses the 22 inputs. It is trained on the 2025 season
and scored on 2026 without refitting.

## Results (2026 season, blind)

| Inputs | ROC-AUC | PR-AUC | Fires inside top 10% of area | Range-day ρ |
|---|---:|---:|---:|---:|
| Terrain only | 0.776 | 0.0065 | 23.7% | 0.237 |
| + Satellite | 0.940 | 0.0372 | 44.4% | 0.749 |
| + Reports | 0.924 | 0.0227 | 29.9% | 0.763 |
| **+ Satellite + Reports** | **0.948** | **0.0406** | **47.9%** | **0.783** |

- *Fires inside top 10%*: if crews cover the 10% of forest flagged each evening,
  this is the share of next-day burning cells they cover. Random would be 10%.
- *Range-day ρ*: rank correlation between expected and observed fire count per
  forest range per day — does the model know which range is in trouble on which day.

The two sources are good at different things. Imagery is better at **where**
(44% vs 30%); the reports are better at **when** (ρ 0.76 vs 0.75, and they
carry the heat spells and rain days the satellite only sees after the fact).
Fused, the model beats both on every measure.

Parser: 505 reports in the 2026 season, about a fifth in Hindi. Every value that
a report states is extracted correctly (`score_extraction` in `textparse.py`).

## The dashboard

**Live site:** `https://<your-github-username>.github.io/<repo-name>/`

- `docs/index.html` is the EcoGuard AI site: the story, what it is, results, the Risk Lab, the build journey and contact.
- `docs/model.html` (same file as `outputs/vanagni_dashboard.html`) is the full model dashboard described below.

Both are self-contained pages. To publish: push this folder to GitHub, then Settings → Pages → Deploy from a branch → `main` and `/docs`.

- Scrub through the season or press Play.
- Switch between the risk map, the Sentinel-2 false-colour scene the model saw,
  and a map of which source drives each cell's risk.
- Rings mark the cells that actually burned that day (hidden from the model).
- Read each range's bulletin with the extracted values highlighted.
- **Edit the bulletin and re-run.** The trained trees are exported to JavaScript,
  so the forecast recomputes in the page. Load the Hindi heatwave sample and watch
  the range light up. `tests/parity.mjs` checks the browser gives exactly the
  Python numbers.

## Environmental Risk Lab (`docs/index.html`, bottom of the page)

The story page ends in a working lab that analyses **your own files, in the browser** (nothing is uploaded):

| Input | What is actually computed |
|---|---|
| PNG / JPG image | colour-rule land cover (vegetation, bare soil, built-up, water, burn-like dark, cloud), VARI greenness, cloud cover, visible fire-coloured pixels |
| GeoTIFF (geotiff.js) | the same, plus true NDVI when a 4th (NIR) band exists; centre coordinates, pixel size and date read from the file |
| Older + newer image | per-pixel class transitions, vegetation loss/gain, new bare soil, new built-up, fragmentation (connected components, edge density), affected area in km² when pixel size is known |
| CSV station data | column detection; max temperature, min humidity, max wind, 7-day rainfall and dry-spell length from the last rows |
| TXT / PDF / pasted text | pattern extraction (English and Hindi) of temperature, humidity, wind speed/direction, rainfall, dry spell; PDF text via pdf.js |

Analyses: wildfire risk, deforestation, urban expansion, wildlife habitat disturbance. Each run shows a six-step
timeline with every parameter's value, unit, source, date and status. Results separate **observed** measurements,
the **estimated** risk level (Low / Moderate / High / Insufficient Data) and **forecasts** (not produced).

It is a **transparent rule-based prototype**: thresholds are listed next to every factor, confidence reflects data
quality only and is capped at Medium because the rules are not validated. Thermal hotspots, terrain and animal
presence are reported as Unavailable because RGB imagery cannot measure them. "Load sample data" fills every
analysis with generated sample files so the lab can be demoed without uploads.

## Run it

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py      # ~90 s: simulate, train, evaluate, export
python scripts/build_dashboard.py   # writes outputs/vanagni_dashboard.html
python -m pytest tests              # parser unit tests
node tests/parity.mjs               # browser engine == Python model
```

Outputs: `metrics.json`, `fused_model.pkl`, `daily_alerts_2026.csv` (what a range
office would receive each evening), and the dashboard.

## About the data

**The study area, imagery, bulletins and fire history in this repo are simulated.**
Building the whole pipeline on a simulator meant we could run it offline, know the
ground truth exactly, and test the parser against known values. The simulator is
deliberately physical rather than random: a daily fuel-moisture bucket per cell,
human ignition concentrated near roads and villages, uphill/wind-driven spread,
Sentinel-2 band responses to drying and to burn scars, realistic cloud cover, and
forecast error in the bulletins. Shapes, units and band names match the real products.

To move to real data, each simulated piece has a real counterpart:

| Here | Real source |
|---|---|
| `synthesise_scenes` | Sentinel-2 L2A from Copernicus Data Space or Google Earth Engine; `imagery.load_geotiff_stack` reads a 4-band GeoTIFF per date |
| VIIRS draws | NASA FIRMS VIIRS 375 m (S-NPP / NOAA-20) |
| `build_terrain` | Copernicus DEM 30 m, ESA WorldCover, OpenStreetMap roads |
| `write_reports` | IMD Dehradun district bulletins; FSI fire alerts; division field reports |
| fire labels | FIRMS detections + FSI burnt-area records |

The parser is the piece most likely to need work on real bulletins; it is kept as
small explicit rules so a forest officer can read and extend it.

## Layout

```
vanagni/
  config.py       study area, seasons, forest ranges
  terrain.py      DEM, slope/aspect, land cover, roads, villages
  weather.py      daily station weather per range (ground truth)
  fire.py         fuel moisture, ignition and spread (ground truth)
  imagery.py      Sentinel-2 bands, cloud mask, NDVI/NDMI/NBR
  bulletins.py    writes English/Hindi bulletins and field notes
  textparse.py    reads them back into numbers, with spans
  features.py     fuses everything into a cell x day table
  model.py        gradient boosting, ablations, attribution, metrics
  export.py       packs the dashboard bundle and exports the trees
scripts/          run_pipeline.py, build_dashboard.py
dashboard/        template.html, engine.js (parser + model in JS)
tests/            parser tests, JS/Python parity
```

## Limits

- One study area, two seasons. Real deployment would train on several years.
- Ignition is human-driven and partly random; no model will catch every fire.
  The honest measure is coverage per unit of patrol effort, which is what we report.
- Attribution is occlusion-based (reset one source to a typical value). It is
  easy to explain, not a full Shapley decomposition.

## Emergency numbers on the site

The "Who to call" section was checked in October 2026 against:
112 / 101 / 108 / 1098 / 14567 / 1033 national helplines; 1077 district control room (district .nic.in pages);
1800 11 9334 MoEFCC helpline for forest fires and poaching (MoEFCC letter, Nov 2022);
Uttarakhand Forest Department fire numbers 1800 180 4141 and 0135 274 4558 (All India Radio, April 2024).
Re-check them before each fire season.

## After the hackathon

See [ROADMAP.md](ROADMAP.md) for the plan to move from simulated to real data and pilot with a forest division.
Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under MIT.
