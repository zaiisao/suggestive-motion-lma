"""Round-2 quick-wins: HistGB hyperparam search, group-aware CV, per-clip
soft-vote aggregation, and a small stacking ensemble.

All experiments reuse the frozen paper manifests so deltas vs. the published
headlines stay apples-to-apples. The new experiments are explicitly labeled
when they change the evaluation protocol (group-aware CV, clip-level acc).
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler

from experiment_quickwins import (
    files_from_manifest, load_raw,
    feat_mean_std, feat_windows_k4, build_matrix,
)


def clip_id_from_path(p):
    """Return the parent dir of an lma_features_*.npy file as the clip id.

    Examples:
      /.../tier0/acting in play/0bdVrgImymc_000020_000030/lma_features_id0.npy
        -> tier0/acting in play/0bdVrgImymc_000020_000030
      /.../tier3/NPDI_features_v2/vid42/lma_features_id7.npy
        -> tier3/NPDI_features_v2/vid42
    The full parent dir uniquely identifies the source video.
    """
    return os.path.dirname(p)


def load_with_clip_ids(files):
    """Like load_raw but also returns parallel list of clip_ids."""
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
        arrs.append(a)
        cids.append(clip_id_from_path(fp))
    return arrs, cids


def build_dataset(manifest_path, drop_tier1, seed, cap=1075):
    tf = files_from_manifest(manifest_path)
    tier_arrs, tier_cids = {}, {}
    for t in (0, 1, 2, 3):
        if not tf[t]:
            continue
        arrs, cids = load_with_clip_ids(tf[t])
        tier_arrs[t], tier_cids[t] = arrs, cids
    if drop_tier1 and 1 in tier_arrs:
        del tier_arrs[1]; del tier_cids[1]

    rng = np.random.RandomState(seed)
    keys = sorted(tier_arrs.keys())
    Xs_ms, Xs_w4, ys, groups = [], [], [], []
    for t in keys:
        arrs, cids = tier_arrs[t], tier_cids[t]
        idx = rng.choice(len(arrs), size=cap, replace=False)
        chosen_arrs = [arrs[i] for i in idx]
        chosen_cids = [cids[i] for i in idx]
        Xms = build_matrix(chosen_arrs, feat_mean_std)
        Xw4 = build_matrix(chosen_arrs, feat_windows_k4)
        Xs_ms.append(Xms); Xs_w4.append(Xw4)
        ys.append(np.full(Xms.shape[0], t))
        groups.extend(chosen_cids)
    X_ms = np.concatenate(Xs_ms); X_w4 = np.concatenate(Xs_w4)
    y = np.concatenate(ys); groups = np.array(groups)
    return X_ms, X_w4, y, groups


def hgb(seed, **kw):
    defaults = dict(max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                    random_state=seed)
    defaults.update(kw)
    return HistGradientBoostingClassifier(**defaults)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--drop-tier1", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paper-acc", type=float, required=True,
                   help="Headline paper accuracy to compare against")
    p.add_argument("--out", default="output/quickwins/round2.csv")
    args = p.parse_args()

    print(f"[*] Loading from manifest: {args.manifest}")
    X_ms, X_w4, y, groups = build_dataset(
        args.manifest, args.drop_tier1, args.seed)
    n_clips = len(set(groups))
    print(f"[+] N fragments = {len(y)}, N clips = {n_clips}")
    print(f"[+] fragments per clip (median / max): "
          f"{int(np.median(np.unique(groups, return_counts=True)[1]))} / "
          f"{int(max(np.unique(groups, return_counts=True)[1]))}")

    # Pre-scale for any linear models we use
    Xms_s = np.nan_to_num(StandardScaler().fit_transform(X_ms))
    Xw4_s = np.nan_to_num(StandardScaler().fit_transform(X_w4))

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_group = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)

    rows = []

    def record(tag, acc, f1, dt, notes=""):
        delta = acc - args.paper_acc
        print(f"   {tag:55s}  acc={acc:.4f}  f1={f1:.4f}  Δ={delta:+.4f}"
              f"  ({dt:.1f}s)  {notes}")
        rows.append(dict(experiment=tag, accuracy=round(acc, 4),
                         f1_macro=round(f1, 4), delta_vs_paper=round(delta, 4),
                         seconds=round(dt, 1), notes=notes))

    # ---- 1. HistGB hyperparameter search on windows_k4, fragment-level ----
    print("\n[1] HistGB hyperparam sweep on windows_k4, fragment-level StratKFold:")
    best = None
    for lr in (0.03, 0.05, 0.1):
        for mi in (300, 600, 1000):
            for ml in (15, 31, 63):
                for l2 in (0.0, 1.0):
                    t0 = time.time()
                    clf = hgb(args.seed, learning_rate=lr, max_iter=mi,
                              max_leaf_nodes=ml, l2_regularization=l2)
                    yp = cross_val_predict(clf, X_w4, y, cv=cv_strat, n_jobs=-1)
                    acc = accuracy_score(y, yp)
                    dt = time.time() - t0
                    tag = f"HGB(lr={lr}, iters={mi}, leaves={ml}, l2={l2})"
                    if best is None or acc > best[0]:
                        best = (acc, tag, dt, yp, clf)
                    print(f"     {tag:55s}  acc={acc:.4f}  ({dt:.1f}s)")
    best_acc, best_tag, best_dt, best_yp, best_clf = best
    f1 = f1_score(y, best_yp, average="macro")
    record(f"BEST hgb(windows_k4) [{best_tag}]", best_acc, f1, best_dt,
           notes="fragment-level, StratifiedKFold")

    # ---- 2. Group-aware CV honesty check ----
    print("\n[2] Same best HGB but with StratifiedGroupKFold by clip_id:")
    t0 = time.time()
    yp_g = cross_val_predict(best_clf, X_w4, y, cv=cv_group, groups=groups,
                             n_jobs=-1)
    acc_g = accuracy_score(y, yp_g); f1_g = f1_score(y, yp_g, average="macro")
    record(f"hgb(windows_k4) groupCV", acc_g, f1_g, time.time() - t0,
           notes="fragment-level, GroupKFold by clip")

    # ---- 3. Per-clip soft-vote aggregation under group CV ----
    print("\n[3] Per-clip soft-vote aggregation, group CV:")
    from sklearn.base import clone
    yp_proba = np.zeros((len(y), len(np.unique(y))))
    for tr, te in cv_group.split(X_w4, y, groups=groups):
        m = clone(best_clf); m.fit(X_w4[tr], y[tr])
        yp_proba[te] = m.predict_proba(X_w4[te])
    classes = sorted(np.unique(y))
    # aggregate per clip in test split (clips are now disjoint across folds)
    clip_to_idx = defaultdict(list)
    for i, g in enumerate(groups):
        clip_to_idx[g].append(i)
    y_true_clip, y_pred_clip = [], []
    for g, idxs in clip_to_idx.items():
        true_lbl = y[idxs[0]]  # all fragments of one clip share label
        # majority among per-fragment soft-votes
        avg_proba = yp_proba[idxs].mean(axis=0)
        pred = classes[int(np.argmax(avg_proba))]
        y_true_clip.append(true_lbl); y_pred_clip.append(pred)
    y_true_clip = np.array(y_true_clip); y_pred_clip = np.array(y_pred_clip)
    acc_c = accuracy_score(y_true_clip, y_pred_clip)
    f1_c = f1_score(y_true_clip, y_pred_clip, average="macro")
    record(f"hgb(windows_k4) groupCV + per-clip soft-vote", acc_c, f1_c, 0,
           notes=f"clip-level, N={len(y_true_clip)} clips")

    # Also show the per-class clip-level confusion matrix
    cm = confusion_matrix(y_true_clip, y_pred_clip, labels=classes)
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    print("\n   Per-clip confusion matrix (rows=true, cols=pred):")
    print("        " + "   ".join(f"T{c}" for c in classes))
    for i, c in enumerate(classes):
        print(f"   T{c}:  " + "  ".join(f"{cm_n[i, j]*100:5.1f}"
                                         for j in range(len(classes))))

    # ---- 4. Stacking: average soft-vote of HGB(mean_std) + HGB(windows_k4) ----
    print("\n[4] Soft-vote ensemble [HGB(mean_std) + HGB(windows_k4)], StratKFold:")
    t0 = time.time()
    clf_ms = hgb(args.seed)
    clf_w4 = best_clf
    proba_ms = np.zeros((len(y), len(classes)))
    proba_w4 = np.zeros((len(y), len(classes)))
    for tr, te in cv_strat.split(X_ms, y):
        m1 = clone(clf_ms); m1.fit(X_ms[tr], y[tr]); proba_ms[te] = m1.predict_proba(X_ms[te])
        m2 = clone(clf_w4); m2.fit(X_w4[tr], y[tr]); proba_w4[te] = m2.predict_proba(X_w4[te])
    proba_avg = (proba_ms + proba_w4) / 2
    yp_ens = np.array([classes[int(i)] for i in np.argmax(proba_avg, axis=1)])
    acc_e = accuracy_score(y, yp_ens); f1_e = f1_score(y, yp_ens, average="macro")
    record(f"ensemble[HGB(mean_std)+HGB(windows_k4)]", acc_e, f1_e,
           time.time() - t0, notes="fragment-level, StratifiedKFold")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[+] wrote {args.out}")


if __name__ == "__main__":
    main()
