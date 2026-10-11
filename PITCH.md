# Vanagni — pitch notes and demo script

## One line

Every evening, tell a forest division which 300 m patches of pine are most likely
to burn tomorrow, by reading the satellite and the weather bulletin together.

## The problem (30 s)

- Uttarakhand's chir pine belt burns every March–June. Most fires are lit by people
  and run through dry needle litter.
- Crews are few. A DFO has to decide each evening where to send them.
- The information exists but sits in different places: satellite scenes in one
  portal, IMD bulletins as text (often Hindi), range officers' notes on paper or
  WhatsApp. Nobody joins them up at cell level.

## Demo (2 min) — follow this order

1. **Open on the peak day** (the dashboard does this). "This is 20 May 2026.
   The model was trained on 2025 and has never seen this season."
2. **Point at the rings.** "These are the cells that actually burned that day.
   The model did not see them. Look where they sit relative to the red."
3. **Switch to Sentinel-2.** "This is the scene the model saw: green is healthy
   forest, the dark red patches are last week's burn scars."
4. **Read the bulletin panel.** "Every highlighted phrase is a number the system
   pulled out of the text: tomorrow's max, humidity, wind, dry spell, danger level.
   Some ranges get Hindi bulletins; same parser."
5. **The moment:** click *Load heatwave (Hindi)*, then *Re-run forecast*.
   "I've just rewritten tomorrow's bulletin as a 40 °C, 11% humidity heatwave.
   The trees run in the browser, so the map recomputes from the text — expected
   fire cells in this range roughly double."
6. **What drives it.** "For each cell, which source is pushing risk up: terrain,
   satellite or the reports. You can see the reports dominate on the heat days."
7. **Scroll to the table.** "Satellite alone is good at *where*, reports alone at
   *when*. Fused beats both on every measure. If crews cover the 10% of forest we
   flag, they cover about 48% of tomorrow's fire — 4.8 times chance."

## Why it should win

- **Genuinely multimodal.** Text and imagery are both necessary, and we prove it with
  an ablation instead of asserting it.
- **Honest evaluation.** Trained on one season, tested blind on the next. Metrics a
  forest officer understands (coverage per patrol effort), not only AUC.
- **Bilingual text understanding** for real Indian bulletins, with every extracted
  value traceable to its sentence.
- **It runs.** One command builds everything in ~90 s on a laptop; the dashboard
  is a single file; the browser engine is bit-for-bit identical to Python (tested).
- **Explainable** per cell, per source.

## Questions judges will ask

**Is this real data?**
No — and say so up front. The area, imagery, bulletins and fires are simulated so we
could know the ground truth and test the parser exactly. The simulator is physical
(fuel moisture, human ignition near roads, wind/slope spread, Sentinel-2 band
responses, cloud, forecast error) and every input has a named real source it maps
to (README, "About the data"). The pipeline already reads real Sentinel-2 GeoTIFFs.

**Why rules for the text instead of an LLM?**
Bulletin language is narrow and stable; rules are auditable, run offline on a
district office laptop, and give exact character spans. An LLM extractor is a
drop-in replacement for `parse()` if the real bulletins turn out messier.

**Why gradient boosting and not a CNN on the imagery?**
Twenty-two meaningful inputs per cell, a few thousand positive examples per season,
and the need to explain each prediction. Trees handle missing values (cloudy pixels,
fields a bulletin skips) natively and run in the browser.

**Couldn't the satellite alone do this?**
The table answers it: satellite-only misses the timing. A heat spell forecast in
tonight's bulletin shows up in the imagery only after the forest has dried.

**What would you do with real deployment?**
Pull Sentinel-2 and FIRMS automatically, scrape the IMD Dehradun bulletin and FSI
alerts each evening, train on 5+ seasons, and send each range office its top-risk
cells as a WhatsApp/SMS list (the `daily_alerts_2026.csv` is that list).

**Accuracy looks low (PR-AUC 0.04)?**
Base rate is 0.28% of cell-days, so 0.04 is about 14× better than chance. The
patrol-coverage number is the one that matters operationally.
