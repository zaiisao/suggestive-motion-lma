"""Post-acceptance quick-win sweep on the 3-way LMA baseline.

Loads the frozen 3-way paper manifest (manifest_paper_3way_2026-04-14_19-16.csv)
so every variant is evaluated on the same fragments. Holds the random seed and
5-fold StratifiedKFold protocol fixed at the paper's values so deltas vs. the
72.06% headline are apples-to-apples.

Variants explored:
  Feature builders   : mean_std (paper) | moments (mean+std+median+iqr+skew)
                      | windows_k4 (4-window mean+std) | windows_k4_moments
  Classifiers        : logreg (paper default) | logreg_tuned (C grid, inner CV)
                      | histgb (HistGradientBoosting defaults)

Run:
  python scripts/experiment_quickwins.py \
    --manifest data/manifest_paper_3way_2026-04-14_19-16.csv \
    --drop-tier1 --out output/quickwins_3way.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import skew

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------- data loading ----------
def files_from_manifest(path):
    out = {0: [], 1: [], 2: [], 3: []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["tier"])].append(row["path"])
    for t in out:
        out[t].sort()
    return out


def load_raw(files):
    """Return list of (T, 55) arrays that pass the paper's filters.

    Matches analyze_lma_tiers.py's aggregate_fragment: requires ndim==2,
    55 features, T>=5, AND the mean+std vector must be all-finite (drops
    fragments whose baseline 110-d vector would contain NaN/inf, exactly
    as the paper pipeline does).
    """
    arrs = []
    for fp in files:
        try:
            a = np.load(fp, allow_pickle=True)
        except Exception:
            continue
        if a.ndim != 2 or a.shape[1] != 55 or a.shape[0] < 5:
            continue
        base = np.concatenate([a.mean(0), a.std(0)])
        if not np.all(np.isfinite(base)):
            continue
        arrs.append(a)
    return arrs


# ---------- feature builders ----------
def feat_mean_std(a):
    return np.concatenate([a.mean(0), a.std(0)])  # 110


def feat_moments(a):
    mu = a.mean(0)
    sd = a.std(0)
    med = np.median(a, axis=0)
    q75, q25 = np.percentile(a, [75, 25], axis=0)
    iqr = q75 - q25
    if a.shape[0] >= 3:
        sk = skew(a, axis=0, bias=False, nan_policy="omit")
        sk = np.nan_to_num(sk, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        sk = np.zeros_like(mu)
    return np.concatenate([mu, sd, med, iqr, sk])  # 275


def _split_windows(a, k):
    n = a.shape[0]
    edges = np.linspace(0, n, k + 1, dtype=int)
    out = []
    for i in range(k):
        s, e = edges[i], edges[i + 1]
        if e - s < 2:  # too small; pad with whole-fragment stat
            out.append(a)
        else:
            out.append(a[s:e])
    return out


def feat_windows_k4(a):
    return np.concatenate([feat_mean_std(w) for w in _split_windows(a, 4)])  # 440


def feat_windows_k4_moments(a):
    return np.concatenate([feat_moments(w) for w in _split_windows(a, 4)])  # 1100


FEATURE_BUILDERS = {
    "mean_std":            feat_mean_std,
    "moments":             feat_moments,
    "windows_k4":          feat_windows_k4,
    "windows_k4_moments":  feat_windows_k4_moments,
}


def build_matrix(arrs, builder):
    rows = []
    for a in arrs:
        v = builder(a)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append(v)
    return np.stack(rows, axis=0)


# ---------- classifiers ----------
# Note: X is pre-scaled once on full data in run() to mirror the paper's
# `analyze_lma_tiers.py` protocol (which has scaler leakage between folds, but
# we keep the exact protocol so deltas vs the 0.7206 headline are comparable).
def make_clf(name, seed):
    if name == "logreg":
        return LogisticRegression(max_iter=2000)
    if name == "logreg_tuned":
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        return GridSearchCV(
            LogisticRegression(max_iter=4000, solver="liblinear"),
            param_grid={"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2", "l1"]},
            cv=inner_cv, n_jobs=-1, scoring="accuracy",
        )
    if name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=None,
            l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(name)


CLFS = ["logreg", "logreg_tuned", "histgb"]


# ---------- experiment driver ----------
def run(args):
    np.random.seed(args.seed)

    tf = files_from_manifest(args.manifest)
    print(f"[*] Loading raw fragments from manifest: {args.manifest}")
    tier_arrs = {}
    for t in (0, 1, 2, 3):
        if not tf[t]:
            continue
        arrs = load_raw(tf[t])
        print(f"    tier{t}: {len(tf[t])} files -> {len(arrs)} valid frags")
        tier_arrs[t] = arrs

    if args.drop_tier1 and 1 in tier_arrs:
        del tier_arrs[1]
        print("[+] Dropped tier1 (3-way mode)")

    # Same balancing semantics as analyze_lma_tiers.py
    min_n = min(len(v) for v in tier_arrs.values())
    cap = min(args.max_per_tier, min_n)
    print(f"[*] Balancing to {cap} fragments per class")

    # IMPORTANT: pick fragment indices once (shared across all feature builders)
    # so every variant sees the SAME fragments, matching paper protocol.
    # Iterate in the same order analyze_lma_tiers.py does (after drop/merge), so
    # np.random.choice consumes RNG identically and we reproduce paper splits.
    balanced_idx = {}
    for t in sorted(tier_arrs.keys()):
        idx = np.random.choice(len(tier_arrs[t]), size=cap, replace=False)
        balanced_idx[t] = idx

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    rows = []
    for fname, builder in FEATURE_BUILDERS.items():
        # Build the matrix once for this feature scheme
        Xs, ys = [], []
        for t, arrs in tier_arrs.items():
            chosen = [arrs[i] for i in balanced_idx[t]]
            Xt = build_matrix(chosen, builder)
            ys.append(np.full(Xt.shape[0], t))
            Xs.append(Xt)
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        # Paper protocol: scale once on full data, then nan-clean.
        Xs_full = StandardScaler().fit_transform(X)
        Xs_full = np.nan_to_num(Xs_full, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"\n=== feature: {fname:20s}  X={X.shape} ===")

        for cname in CLFS:
            t0 = time.time()
            clf = make_clf(cname, args.seed)
            X_in = X if cname == "histgb" else Xs_full  # trees don't need scale
            y_pred = cross_val_predict(clf, X_in, y, cv=cv, n_jobs=-1)
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average="macro")
            dt = time.time() - t0
            print(f"   {cname:14s}  acc={acc:.4f}  f1={f1:.4f}  ({dt:.1f}s)")
            rows.append({
                "feature": fname, "classifier": cname,
                "dim": int(X.shape[1]),
                "accuracy": round(acc, 4),
                "f1_macro": round(f1, 4),
                "seconds": round(dt, 1),
            })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    paper = 0.7206
    df["delta_vs_paper"] = (df["accuracy"] - paper).round(4)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n[+] wrote {args.out}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--drop-tier1", action="store_true")
    p.add_argument("--max-per-tier", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="output/quickwins/quickwins_3way.csv")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
