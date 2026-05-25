"""Binary SFW/NSFW: round-3 recipe (uniform avg of HGB on ms+w4+py features,
clip-level soft-vote under StratifiedGroupKFold).

Merges {T0, T1} -> SFW and {T2, T3} -> NSFW per the paper's --binary mode.
Paper baseline: 0.7869 (mean_std + LogReg, fragment-level StratKFold).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, cross_val_predict,
)
from sklearn.preprocessing import StandardScaler

from experiment_quickwins import (
    files_from_manifest, feat_mean_std, build_matrix,
)
from experiment_cliplevel import load_with_clip_ids
from experiment_round3 import feat_windows_k4, feat_pyramid, hgb_tuned


def build_binary_dataset(manifest_path, seed, max_per_class=2000):
    """Load 4 tiers, merge {0,1}->SFW, {2,3}->NSFW, balance."""
    tf = files_from_manifest(manifest_path)
    raw_arrs, raw_cids = {}, {}
    for t in (0, 1, 2, 3):
        if not tf[t]:
            raw_arrs[t], raw_cids[t] = [], []
            continue
        a, c = load_with_clip_ids(tf[t])
        raw_arrs[t], raw_cids[t] = a, c
        print(f"    tier{t}: {len(tf[t])} files -> {len(a)} valid frags")

    sfw_arrs = raw_arrs[0] + raw_arrs[1]
    sfw_cids = raw_cids[0] + raw_cids[1]
    nsfw_arrs = raw_arrs[2] + raw_arrs[3]
    nsfw_cids = raw_cids[2] + raw_cids[3]
    # Also track origin tier for per-class breakdown
    sfw_origin = ([0] * len(raw_arrs[0])) + ([1] * len(raw_arrs[1]))
    nsfw_origin = ([2] * len(raw_arrs[2])) + ([3] * len(raw_arrs[3]))

    cap = min(max_per_class, len(sfw_arrs), len(nsfw_arrs))
    print(f"[*] SFW={len(sfw_arrs)}, NSFW={len(nsfw_arrs)} -> cap {cap} per class")

    rng = np.random.RandomState(seed)
    pieces = []
    for label, arrs, cids, origins in [
        (0, sfw_arrs, sfw_cids, sfw_origin),
        (1, nsfw_arrs, nsfw_cids, nsfw_origin),
    ]:
        idx = rng.choice(len(arrs), size=cap, replace=False)
        ca = [arrs[i] for i in idx]
        cc = [cids[i] for i in idx]
        co = [origins[i] for i in idx]
        Xms = build_matrix(ca, feat_mean_std)
        Xw4 = build_matrix(ca, feat_windows_k4)
        Xpy = build_matrix(ca, feat_pyramid)
        pieces.append((Xms, Xw4, Xpy, np.full(cap, label), np.array(cc),
                       np.array(co)))

    X_ms = np.concatenate([p[0] for p in pieces])
    X_w4 = np.concatenate([p[1] for p in pieces])
    X_py = np.concatenate([p[2] for p in pieces])
    y    = np.concatenate([p[3] for p in pieces])
    groups = np.concatenate([p[4] for p in pieces])
    origin = np.concatenate([p[5] for p in pieces])
    return X_ms, X_w4, X_py, y, groups, origin


def per_origin_recall(y_true_clip, y_pred_clip, origin_clip):
    """Recall per original tier (0,1,2,3) on clip predictions.

    SFW correct = predicted 0, NSFW correct = predicted 1.
    """
    out = {}
    for t in (0, 1, 2, 3):
        m = origin_clip == t
        if m.sum() == 0:
            continue
        if t in (0, 1):
            correct = (y_pred_clip[m] == 0).mean()
        else:
            correct = (y_pred_clip[m] == 1).mean()
        out[f"T{t}"] = round(float(correct), 4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",
                   default="data/manifest_paper_binary_2026-04-14_21-36.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-class", type=int, default=2000,
                   help="paper used --max-per-tier 2000")
    p.add_argument("--paper-acc", type=float, default=0.7869)
    p.add_argument("--out", default="output/quickwins/binary_round3.csv")
    args = p.parse_args()

    print(f"[*] Loading {args.manifest}")
    X_ms, X_w4, X_py, y, groups, origin = build_binary_dataset(
        args.manifest, args.seed, args.max_per_class)
    n_clips = len(set(groups))
    print(f"[+] N frags={len(y)}, N clips={n_clips}")
    print(f"[+] dim: mean_std={X_ms.shape[1]}, windows_k4={X_w4.shape[1]}, pyramid={X_py.shape[1]}")

    classes = np.array([0, 1])
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_group = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)

    rows = []

    # ---- paper baseline reproduction ----
    print("\n[1] paper baseline (mean_std + LogReg, fragment StratKF)")
    t0 = time.time()
    Xms_s = np.nan_to_num(StandardScaler().fit_transform(X_ms))
    yp = cross_val_predict(LogisticRegression(max_iter=2000), Xms_s, y,
                            cv=cv_strat, n_jobs=4)
    acc = accuracy_score(y, yp); f1 = f1_score(y, yp, average="macro")
    print(f"  acc={acc:.4f}  f1={f1:.4f}  Δ={acc-args.paper_acc:+.4f}  ({time.time()-t0:.1f}s)")
    rows.append(dict(experiment="paper_baseline (frag, StratKF)", acc=round(acc, 4),
                     f1=round(f1, 4), delta=round(acc - args.paper_acc, 4)))

    # ---- best single: pyramid HGB clip-level GroupKF ----
    print("\n[2] pyramid HGB, GroupKF + clip-level soft-vote")
    t0 = time.time()
    proba = np.zeros((len(y), len(classes)))
    for fi, (tr, te) in enumerate(cv_group.split(X_py, y, groups=groups)):
        m = hgb_tuned(args.seed); m.fit(X_py[tr], y[tr])
        proba[te] = m.predict_proba(X_py[te])
        print(f"     fold {fi+1}/5 done")
    # clip-level aggregation
    clip_to_idx = defaultdict(list)
    for i, g in enumerate(groups):
        clip_to_idx[g].append(i)
    yt_c, yp_c, or_c = [], [], []
    for g, idxs in clip_to_idx.items():
        yt_c.append(y[idxs[0]])
        yp_c.append(classes[int(np.argmax(proba[idxs].mean(0)))])
        or_c.append(origin[idxs[0]])
    yt_c = np.array(yt_c); yp_c = np.array(yp_c); or_c = np.array(or_c)
    acc_c = accuracy_score(yt_c, yp_c); f1_c = f1_score(yt_c, yp_c, average="macro")
    print(f"  clip-level acc={acc_c:.4f}  f1={f1_c:.4f}  Δ={acc_c-args.paper_acc:+.4f}  "
          f"({len(yt_c)} clips, {time.time()-t0:.1f}s)")
    rows.append(dict(experiment="pyramid clip-softvote (GroupKF)", acc=round(acc_c, 4),
                     f1=round(f1_c, 4), delta=round(acc_c - args.paper_acc, 4)))
    per_orig = per_origin_recall(yt_c, yp_c, or_c)
    print(f"  per-origin clip recall: {per_orig}")

    # ---- round-3 best: uniform avg of HGB(ms+w4+py), clip GroupKF ----
    print("\n[3] uniform avg HGB(ms+w4+py), GroupKF + clip-level soft-vote")
    t0 = time.time()
    proba_all = [np.zeros((len(y), len(classes))) for _ in range(3)]
    for fi, (tr, te) in enumerate(cv_group.split(X_ms, y, groups=groups)):
        for bi, Xb in enumerate([X_ms, X_w4, X_py]):
            m = hgb_tuned(args.seed); m.fit(Xb[tr], y[tr])
            proba_all[bi][te] = m.predict_proba(Xb[te])
        print(f"     fold {fi+1}/5 done")
    proba_avg = sum(proba_all) / 3
    yt_c, yp_c, or_c = [], [], []
    for g, idxs in clip_to_idx.items():
        yt_c.append(y[idxs[0]])
        yp_c.append(classes[int(np.argmax(proba_avg[idxs].mean(0)))])
        or_c.append(origin[idxs[0]])
    yt_c = np.array(yt_c); yp_c = np.array(yp_c); or_c = np.array(or_c)
    acc_c = accuracy_score(yt_c, yp_c); f1_c = f1_score(yt_c, yp_c, average="macro")
    print(f"  clip-level acc={acc_c:.4f}  f1={f1_c:.4f}  Δ={acc_c-args.paper_acc:+.4f}  ({time.time()-t0:.1f}s)")
    rows.append(dict(experiment="uniform_avg3 clip-softvote (GroupKF)", acc=round(acc_c, 4),
                     f1=round(f1_c, 4), delta=round(acc_c - args.paper_acc, 4)))
    per_orig = per_origin_recall(yt_c, yp_c, or_c)
    print(f"  per-origin clip recall (best): {per_orig}")

    cm = confusion_matrix(yt_c, yp_c, labels=classes)
    cm_n = cm / cm.sum(1, keepdims=True)
    print("\n  Clip-level binary confusion (rows=true, cols=pred):")
    print("           SFW     NSFW")
    for i, name in enumerate(("SFW", "NSFW")):
        print(f"   {name:4s}:  {cm_n[i,0]*100:5.1f}    {cm_n[i,1]*100:5.1f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[+] wrote {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
