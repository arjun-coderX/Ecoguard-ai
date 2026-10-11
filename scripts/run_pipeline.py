"""End-to-end run: simulate two seasons, train on 2025, test blind on 2026,
run ablations, and write the dashboard bundle.

    python scripts/run_pipeline.py            # ~1 minute on a laptop
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vanagni.config import CFG                                    # noqa: E402
from vanagni.export import export_dashboard                        # noqa: E402
from vanagni.features import ALL, simulate, season_table, day_features  # noqa: E402
from vanagni.model import (RiskModel, VARIANTS, evaluate,          # noqa: E402
                           permutation_importance)
from vanagni.terrain import build_terrain                          # noqa: E402
from vanagni.textparse import score_extraction                     # noqa: E402

NEG_RATE = 0.08
OUT = ROOT / "outputs"


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:5.1f}s] {msg}", flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    terrain = build_terrain()
    log("terrain built")
    train = simulate(terrain, CFG.train_season)
    test = simulate(terrain, CFG.test_season)
    log(f"seasons simulated: {train.burning.sum()} / {test.burning.sum()} burning cell-days")

    Xtr, ytr, _, _ = season_table(terrain, train, NEG_RATE)
    Xte, yte, dte, cte = season_table(terrain, test)
    rte = terrain.range_id.reshape(-1)[cte]
    log(f"features: train {Xtr.shape}, test {Xte.shape}")

    results, models = {}, {}
    for name, feats in VARIANTS.items():
        m = RiskModel(feats, NEG_RATE).fit(Xtr, ytr)
        results[name] = evaluate(m.predict(Xte), yte, dte, rte)
        models[name] = m
        r = results[name]
        log(f"{name:15s} ROC {r['roc_auc']:.3f}  PR {r['pr_auc']:.4f}  "
            f"top10% {r['capture_top10']:.1%}  range-day rho {r['range_day_rho']:.3f}")

    fused = models["fused"]
    rng = np.random.default_rng(0)
    sample = rng.choice(len(yte), 250_000, replace=False)
    sample = np.union1d(sample, np.flatnonzero(yte))
    imp = permutation_importance(fused, Xte[sample], yte[sample], dte[sample], rng)
    log("permutation importance done")

    parser = score_extraction(test.docs)
    attr_baseline = np.nanmedian(Xtr, axis=0)

    metrics = dict(
        train_season=CFG.train_season, test_season=CFG.test_season,
        train_fires=int(train.burning.sum()), test_fires=int(test.burning.sum()),
        n_reports=len(test.docs),
        hindi_share=float(np.mean([d["lang"] == "hi" for d in test.docs])),
        ablation=results, importance=imp, parser=parser,
    )
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(OUT / "fused_model.pkl", "wb") as fh:
        pickle.dump(fused, fh)

    # parity fixture: Python predictions for one peak day, checked from JS
    check_day = int(np.argmax(test.burning.sum((1, 2))))
    Xd, _ = day_features(terrain, test, check_day)
    np.save(OUT / "parity_day.npy", np.array([check_day]))
    np.save(OUT / "parity_pred.npy", fused.predict(Xd).astype(np.float32))

    # daily alert sheet a range office could actually receive, plus the
    # per-range series the dashboard timeline draws
    from vanagni.bulletins import season_date
    rows = ["date,range,expected_fire_cells,observed_fire_cells,top_risk_cell_pct"]
    rid = terrain.range_id.reshape(-1)
    R = len(CFG.ranges)
    series = dict(expected=[[0.0] * CFG.n_days for _ in range(R)],
                  observed=[[0] * CFG.n_days for _ in range(R)])
    all_p = []
    for d in range(1, CFG.n_days):
        X, y = day_features(terrain, test, d)
        p = fused.predict(X)
        all_p.append(p)
        for r, (name, *_) in enumerate(CFG.ranges):
            m = rid == r
            series["expected"][r][d] = round(float(p[m].sum()), 3)
            series["observed"][r][d] = int(y[m].sum())
            rows.append(f"{season_date(test.year, d)},{name},{p[m].sum():.2f},"
                        f"{int(y[m].sum())},{100 * p[m].max():.2f}")
    (OUT / "daily_alerts_2026.csv").write_text("\n".join(rows))
    q = np.quantile(np.concatenate(all_p), [0.5, 0.8, 0.9, 0.95, 0.99, 0.999])
    log("risk quantiles 50/80/90/95/99/99.9%: " + " ".join(f"{v:.5f}" for v in q))
    metrics["series"] = series

    export_dashboard(OUT / "dashboard_data.json", terrain, test, fused,
                     attr_baseline, metrics)
    log("dashboard bundle written")
    log("done")


if __name__ == "__main__":
    main()
