# Contributing

Thanks for wanting to help. Small, focused pull requests are easiest to review.

1. `pip install -r requirements.txt`
2. Run `python -m pytest tests` and `node tests/parity.mjs` before and after your change.
3. The site lives in `dashboard/scenarios.html` (copied to `docs/index.html` for GitHub Pages). The full model dashboard is built by `python scripts/build_dashboard.py`.
4. Never present simulated or sample numbers as real. If a value can't be measured, show "Unavailable".
5. Emergency numbers: only add numbers you have checked on an official (.gov.in / .nic.in) page, and note the date.

Good first issues: new bulletin phrasings for `textparse.py`, more Hindi patterns, state forest-fire helplines in the "Who to call" section.

Contact: Arjun Ray · arjun.onlywork@gmail.com
