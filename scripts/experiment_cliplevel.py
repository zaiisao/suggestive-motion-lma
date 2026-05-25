"""Standalone clip-level + group-CV + ensemble experiment.

Uses the round-1 best HistGB config (sklearn defaults plus lr=0.05, l2=1.0,
max_iter=400) so it can run in parallel with the round-2 hyperparam sweep
without depending on its results.

Reports four numbers:
  1. fragment-level, StratifiedKFold        (paper-comparable)
  2. fragment-level, StratifiedGroupKFold   (honesty check: same clip not split)
  3. clip-level, StratifiedGroupKFold + soft-vote
  4. fragment-level, StratKFold, ensemble [HGB(mean_std) + HGB(windows_k4)]
"""
import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, cross_val_predict,
)
from sklearn.preprocessing import StandardScaler

from experiment_quickwins import (
    files_from_manifest, feat_mean_std, feat_windows_k4, build_matrix,
)


def clip_id_from_path(p):
    return os.path.dirname(p)


def load_with_clip_ids(files):
    arrs, cids = [], []
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
        arrs.append(a); cids.append(clip_id_from_path(fp))
    return arrs, cids


def build_dataset(manifest_path, drop_tier1, seed, cap=1075):
    tf = files_from_manifest(manifest_path)
    tier_arrs, tier_cids = {}, {}
    for t in (0, 1, 2, 3):
        if not tf[t]:
            continue
        a, c = load_with_clip_ids(tf[t])
        tier_arrs[t], tier_cids[t] = a, c
    if drop_tier1 and 1 in tier_arrs:
        del tier_arrs[1]; del tier_cids[1]

    rng = np.random.RandomState(seed)
    Xms, Xw4, ys, groups = [], [], [], []
    for t in sorted(tier_arrs.keys()):
        arrs, cids = tier_arrs[t], tier_cids[t]
        idx = rng.choice(len(arrs), size=cap, replace=False)
        ca = [arrs[i] for i in idx]
        cc = [cids[i] for i in idx]
        Xms.append(build_matrix(ca, feat_mean_std))
        Xw4.append(build_matrix(ca, feat_windows_k4))
        ys.append(np.full(cap, t))
        groups.extend(cc)
    return (np.concatenate(Xms), np.concatenate(Xw4),
            np.concatenate(ys), np.array(groups))


def hgb(seed):
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, l2_regularization=1.0,
        random_state=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--drop-tier1", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paper-acc", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-jobs", type=int, default=4,
                   help="cap parallelism so we don't fight the round-2 sweep")
    args = p.parse_args()

    print(f"[*] Loading {args.manifest}")
    X_ms, X_w4, y, groups = build_dataset(
        args.manifest, args.drop_tier1, args.seed)
    cnt = np.unique(groups, return_counts=True)[1]
    print(f"[+] N frags={len(y)}, N clips={len(cnt)}, "
          f"median frags/clip={int(np.median(cnt))}, max={int(cnt.max())}")
    classes = sorted(np.unique(y))

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_group = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                    random_state=args.seed)

    rows = []
    def add(tag, acc, f1, dt, notes=""):
        d = acc - args.paper_acc
        print(f"   {tag:60s}  acc={acc:.4f}  f1={f1:.4f}  Δ={d:+.4f}"
              f"  ({dt:.1f}s)  {notes}")
        rows.append(dict(experiment=tag, accuracy=round(acc, 4),
                         f1_macro=round(f1, 4),
                         delta_vs_paper=round(d, 4),
                         seconds=round(dt, 1), notes=notes))

    # 1. Fragment-level, StratKFold (paper-comparable)
    print("\n[1] fragment-level, StratKFold")
    t0 = time.time()
    yp = cross_val_predict(hgb(args.seed), X_w4, y, cv=cv_strat,
                           n_jobs=args.n_jobs)
    add("hgb(windows_k4) frag/StratKF",
        accuracy_score(y, yp), f1_score(y, yp, average="macro"),
        time.time() - t0)

    # 2. Fragment-level, GroupKFold (honesty check)
    print("\n[2] fragment-level, GroupKFold by clip")
    t0 = time.time()
    yp_g = cross_val_predict(hgb(args.seed), X_w4, y, cv=cv_group,
                             groups=groups, n_jobs=args.n_jobs)
    add("hgb(windows_k4) frag/GroupKF",
        accuracy_score(y, yp_g), f1_score(y, yp_g, average="macro"),
        time.time() - t0, notes="no clip-leakage")

    # 3. Clip-level, GroupKFold + soft-vote
    print("\n[3] clip-level soft-vote, GroupKFold")
    t0 = time.time()
    proba = np.zeros((len(y), len(classes)))
    for tr, te in cv_group.split(X_w4, y, groups=groups):
        m = clone(hgb(args.seed)); m.fit(X_w4[tr], y[tr])
        proba[te] = m.predict_proba(X_w4[te])
    clip_to_idx = defaultdict(list)
    for i, g in enumerate(groups):
        clip_to_idx[g].append(i)
    y_true_c, y_pred_c = [], []
    for g, idxs in clip_to_idx.items():
        y_true_c.append(y[idxs[0]])
        y_pred_c.append(classes[int(np.argmax(proba[idxs].mean(0)))])
    y_true_c = np.array(y_true_c); y_pred_c = np.array(y_pred_c)
    add(f"hgb(windows_k4) clip-softvote/GroupKF",
        accuracy_score(y_true_c, y_pred_c),
        f1_score(y_true_c, y_pred_c, average="macro"),
        time.time() - t0, notes=f"{len(y_true_c)} clips")
    cm = confusion_matrix(y_true_c, y_pred_c, labels=classes)
    cm_n = cm / cm.sum(1, keepdims=True)
    print("\n   Clip-level confusion (rows=true, cols=pred):")
    print("        " + "   ".join(f"T{c}" for c in classes))
    for i, c in enumerate(classes):
        print(f"   T{c}:  " + "  ".join(f"{cm_n[i,j]*100:5.1f}"
                                         for j in range(len(classes))))

    # 4. Ensemble: HGB(mean_std) + HGB(windows_k4), soft-vote average
    print("\n[4] ensemble [hgb(mean_std)+hgb(windows_k4)], StratKFold")
    t0 = time.time()
    p_ms = np.zeros((len(y), len(classes)))
    p_w4 = np.zeros((len(y), len(classes)))
    for tr, te in cv_strat.split(X_ms, y):
        m1 = clone(hgb(args.seed)); m1.fit(X_ms[tr], y[tr])
        p_ms[te] = m1.predict_proba(X_ms[te])
        m2 = clone(hgb(args.seed)); m2.fit(X_w4[tr], y[tr])
        p_w4[te] = m2.predict_proba(X_w4[te])
    p_avg = (p_ms + p_w4) / 2
    yp_e = np.array([classes[int(i)] for i in np.argmax(p_avg, 1)])
    add("ensemble[hgb(ms)+hgb(w4)] frag/StratKF",
        accuracy_score(y, yp_e), f1_score(y, yp_e, average="macro"),
        time.time() - t0)

    # 5. Bonus: per-clip soft-vote of the ensemble under GroupKFold
    print("\n[5] ensemble + per-clip soft-vote, GroupKFold")
    t0 = time.time()
    p_ms_g = np.zeros((len(y), len(classes)))
    p_w4_g = np.zeros((len(y), len(classes)))
    for tr, te in cv_group.split(X_w4, y, groups=groups):
        m1 = clone(hgb(args.seed)); m1.fit(X_ms[tr], y[tr])
        p_ms_g[te] = m1.predict_proba(X_ms[te])
        m2 = clone(hgb(args.seed)); m2.fit(X_w4[tr], y[tr])
        p_w4_g[te] = m2.predict_proba(X_w4[te])
    p_ens_g = (p_ms_g + p_w4_g) / 2
    y_true_c2, y_pred_c2 = [], []
    for g, idxs in clip_to_idx.items():
        y_true_c2.append(y[idxs[0]])
        y_pred_c2.append(classes[int(np.argmax(p_ens_g[idxs].mean(0)))])
    y_true_c2 = np.array(y_true_c2); y_pred_c2 = np.array(y_pred_c2)
    add("ensemble clip-softvote/GroupKF",
        accuracy_score(y_true_c2, y_pred_c2),
        f1_score(y_true_c2, y_pred_c2, average="macro"),
        time.time() - t0)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[+] wrote {args.out}")


if __name__ == "__main__":
    main()
