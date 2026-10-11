"""Gradient-boosted fire-risk model, ablations and attribution.

Trained on one season, scored blind on the next. Negatives are subsampled
for training; predicted probabilities are corrected back to the true base
rate so the numbers on the map are real daily probabilities.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import ALL, CATEGORICAL, TERRAIN, IMAGERY, TEXT

VARIANTS = {
    "terrain only": TERRAIN,
    "satellite only": TERRAIN + IMAGERY,
    "reports only": TERRAIN + TEXT,
    "fused": ALL,
}


class RiskModel:
    def __init__(self, features, neg_rate):
        self.features = features
        self.idx = [ALL.index(f) for f in features]
        self.neg_rate = neg_rate
        cat = [f in CATEGORICAL for f in features]
        self.clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            categorical_features=cat, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=30, random_state=0)

    def fit(self, X, y):
        self.clf.fit(X[:, self.idx], y)
        return self

    def predict(self, X):
        p = self.clf.predict_proba(X[:, self.idx])[:, 1]
        w = self.neg_rate                      # undo negative subsampling
        return p * w / (p * w + (1 - p))


def capture_at(p, y, days, top=0.05):
    """Share of the day's burning cells inside its top-`top` riskiest cells,
    pooled over the season. This is the number a DFO cares about: if I can
    only patrol 5% of the forest tomorrow, how much of the fire do I cover?"""
    caught = total = 0
    for d in np.unique(days):
        m = days == d
        if y[m].sum() == 0:
            continue
        k = max(1, int(round(top * m.sum())))
        thr = np.partition(p[m], -k)[-k]
        caught += (y[m] & (p[m] >= thr)).sum()
        total += y[m].sum()
    return caught / total


def pooled_capture(p, y, top=0.01):
    """Top `top` of all cell-days in the season - rewards getting both the
    place *and* the day right."""
    k = max(1, int(round(top * len(p))))
    thr = np.partition(p, -k)[-k]
    return (y & (p >= thr)).sum() / y.sum()


def range_day_skill(p, y, days, ranges):
    """Rank correlation between expected and observed fire count per
    forest range per day - i.e. does the model know which ranges are in
    trouble on which days."""
    key = days.astype(np.int64) * 100 + ranges
    uniq, inv = np.unique(key, return_inverse=True)
    exp_ = np.bincount(inv, weights=p)
    obs = np.bincount(inv, weights=y.astype(float))
    return float(spearmanr(exp_, obs).statistic)


def evaluate(p, y, days, ranges):
    return dict(roc_auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)),
                capture_top5=float(capture_at(p, y, days, 0.05)),
                capture_top10=float(capture_at(p, y, days, 0.10)),
                pooled_top1=float(pooled_capture(p, y, 0.01)),
                range_day_rho=range_day_skill(p, y, days, ranges),
                base_rate=float(y.mean()))


def group_attribution(model, X, baseline):
    """Occlusion attribution by data source.

    For each cell, swap one source's features for their training-season
    typical values and see how much the predicted risk drops. Cheap, honest,
    and easy to explain to a non-ML audience.
    """
    full = model.predict(X)
    out = {}
    for name, feats in [("terrain", TERRAIN), ("satellite", IMAGERY), ("reports", TEXT)]:
        Xo = X.copy()
        for f in feats:
            Xo[:, ALL.index(f)] = baseline[ALL.index(f)]
        out[name] = full - model.predict(Xo)
    return full, out


def permutation_importance(model, X, y, days, rng, repeats=2):
    """Drop in PR-AUC when each feature is shuffled (on a test sample)."""
    base = average_precision_score(y, model.predict(X))
    res = {}
    for f in model.features:
        j = ALL.index(f)
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - average_precision_score(y, model.predict(Xp)))
        res[f] = float(np.mean(drops))
    return res
