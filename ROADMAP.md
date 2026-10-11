# EcoGuard AI — roadmap after the hackathon

Arjun Ray · arjun.onlywork@gmail.com

What exists today is a working prototype on **simulated** data: the full pipeline (imagery + bulletins → features → model → map), a browser Risk Lab that runs transparent rules on real files, and a story page. The plan below turns it into something a forest division could use for a season.

## Phase 1 — real data, same model (weeks 1–6)

| Today (simulated) | Replace with | How |
|---|---|---|
| `synthesise_scenes` | Sentinel-2 L2A, 10–20 m | Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`) or Copernicus Data Space; export NDVI/NDMI/NBR composites per 5-day window. `imagery.load_geotiff_stack` already reads 4-band GeoTIFFs. |
| VIIRS draws / fire labels | NASA FIRMS VIIRS 375 m | FIRMS area API (free MAP_KEY). Use as training labels and as the "fires nearby last 2 nights" input. |
| `build_terrain` | Copernicus DEM 30 m, ESA WorldCover, OpenStreetMap roads | One-time download for the study area. |
| `write_reports` | IMD district bulletins, FSI fire alerts | Collect PDFs/text daily; the parser (`textparse.py`) is the part most likely to need new rules. Log every value it misses. |
| weather | Open-Meteo or IMD gridded data | Gives a numeric fallback when a bulletin is missing. |

Done when: the 2024 and 2025 Uttarakhand fire seasons are rebuilt from real data and the ablation table is re-run honestly (expect lower numbers than the simulator).

## Phase 2 — a tool for one forest division (weeks 6–12)

- Nightly job (GitHub Actions or a small server) that pulls the day's data, scores every cell and publishes the map.
- Evening alert: top-10% cells per range as a WhatsApp/SMS message and a printable PDF for the range office.
- Feedback button for field staff: "fire here / no fire / wrong place". This becomes new training data.
- Hindi-first interface for the field version.

## Phase 3 — trust and scale

- Calibrate probabilities (reliability plots) before showing any percentage to users.
- Validate against FSI burnt-area records, not only FIRMS.
- Add other states with different fire regimes (central India sal forests, north-east jhum).
- Extend the Risk Lab's rule-based deforestation and urban modules with trained models where labelled data exists.

## Partners to approach

- Uttarakhand Forest Department (a DFO in Dehradun, Mussoorie or Tehri division)
- Forest Survey of India (fire alert system)
- University remote-sensing labs (IIRS Dehradun is next door)
- Student/hackathon mentors for compute credits (Earth Engine is free for research and non-profit use)

## Known limits to keep stating

- Model results so far come from a simulator; they show the method works, not real-world accuracy.
- The Risk Lab uses colour rules on RGB images; it cannot see heat, moisture or terrain without extra bands.
- Fires are mostly lit by people; no model will catch every one. The honest metric is fires covered per unit of patrol effort.
