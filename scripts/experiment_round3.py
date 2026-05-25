"""Round-3: pyramid temporal pooling + stacking meta-learner.

Pyramid features: concat mean+std at K=1, K=2, K=4, K=8 windows  -> 1650-dim.
Stacking: HGB(mean_std), HGB(windows_k4), HGB(pyramid) -> LogReg meta-learner
on out-of-fold proba.

Eval protocols:
  - fragment-level StratKFold        (paper-comparable)
  - fragment-level GroupKFold        (honesty check)
  - clip-level soft-vote, GroupKFold (real-world)
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
    StratifiedKFold, StratifiedGroupKFold,
)
from sklearn.preprocessing import StandardScaler

from experiment_quickwins import (
    files_from_manifest, feat_mean_std, build_matrix, _split_windows,
)
from experiment_cliplevel import load_with_clip_ids


# ---- feature builders ----
def feat_windows_k(a, k):
    return np.concatenate([
        np.concatenate([w.mean(0), w.std(0)]) for w in _split_windows(a, k)
    ])

def feat_pyramid(a):
    """Concat mean+std at K=1, K=2, K=4, K=8 -> 110*(1+2+4+8) = 1650-dim."""
    return np.concatenate([feat_windows_k(a, k) for k in (1, 2, 4, 8)])

def feat_windows_k4(a):
    return feat_windows_k(a, 4)


# ---- dataset assembly ----
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
    Xms, Xw4, Xpy, ys, groups = [], [], [], [], []
    for t in sorted(tier_arrs.keys()):
        arrs, cids = tier_arrs[t], tier_cids[t]
        idx = rng.choice(len(arrs), size=cap, replace=False)
        ca = [arrs[i] for i in idx]
        cc = [cids[i] for i in idx]
        Xms.append(build_matrix(ca, feat_mean_std))
        Xw4.append(build_matrix(ca, feat_windows_k4))
        Xpy.append(build_matrix(ca, feat_pyramid))
        ys.append(np.full(cap, t))
        groups.extend(cc)
    return (np.concatenate(Xms), np.concatenate(Xw4), np.concatenate(Xpy),
            np.concatenate(ys), np.array(groups))


def hgb_tuned(seed):
    """Best config from round-2 sweep."""
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=seed,
    )


# ---- core: stacked predict_proba over outer folds ----
def stacked_proba(X_list, y, cv, groups=None, seed=42, inner_splits=5):
    """For each outer split, fit base learners, generate out-of-fold proba on
    the outer-train via inner CV, train meta-learner, then predict proba on
    outer-test (using base learners fit on the full outer-train).

    Returns (n_samples, n_classes) proba aligned with y, plus the base-model
    contribution proba dict.
    """
    classes = np.array(sorted(np.unique(y)))
    n_cls = len(classes)
    out_proba = np.zeros((len(y), n_cls))
    base_oof = {i: np.zeros((len(y), n_cls)) for i in range(len(X_list))}

    splits = list(cv.split(X_list[0], y, groups=groups)) if groups is not None \
        else list(cv.split(X_list[0], y))

    for fold_i, (tr, te) in enumerate(splits):
        # 1. Inner CV on the outer-train: generate base OOF proba for meta train
        if groups is not None:
            inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True,
                                          random_state=seed + fold_i)
            inner_splits_list = list(inner.split(X_list[0][tr], y[tr],
                                                  groups=groups[tr]))
        else:
            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                    random_state=seed + fold_i)
            inner_splits_list = list(inner.split(X_list[0][tr], y[tr]))

        meta_train = np.zeros((len(tr), len(X_list) * n_cls))
        meta_test = np.zeros((len(te), len(X_list) * n_cls))

        for bi, Xb in enumerate(X_list):
            Xb_tr = Xb[tr]; Xb_te = Xb[te]
            inner_proba = np.zeros((len(tr), n_cls))
            for itr, ite in inner_splits_list:
                m = hgb_tuned(seed + fold_i)
                m.fit(Xb_tr[itr], y[tr][itr])
                inner_proba[ite] = m.predict_proba(Xb_tr[ite])
            # Fit on full outer-train, predict outer-test
            m_full = hgb_tuned(seed + fold_i)
            m_full.fit(Xb_tr, y[tr])
            test_proba = m_full.predict_proba(Xb_te)

            meta_train[:, bi*n_cls:(bi+1)*n_cls] = inner_proba
            meta_test[:, bi*n_cls:(bi+1)*n_cls] = test_proba
            base_oof[bi][te] = test_proba

        # 2. Train meta on inner OOF; predict on outer-test
        meta = LogisticRegression(max_iter=2000, C=1.0)
        meta.fit(meta_train, y[tr])
        out_proba[te] = meta.predict_proba(meta_test)

        print(f"     fold {fold_i+1}/{len(splits)} done")

    return out_proba, base_oof, classes


def evaluate(name, y, proba, classes, groups=None):
    yp = np.array([classes[i] for i in np.argmax(proba, axis=1)])
    acc = accuracy_score(y, yp); f1 = f1_score(y, yp, average="macro")
    print(f"  [{name}] frag-level: acc={acc:.4f} f1={f1:.4f}")

    if groups is None:
        return {"frag_acc": acc, "frag_f1": f1}

    # clip-level soft-vote
    clip_to_idx = defaultdict(list)
    for i, g in enumerate(groups):
        clip_to_idx[g].append(i)
    y_true_c, y_pred_c = [], []
    for g, idxs in clip_to_idx.items():
        y_true_c.append(y[idxs[0]])
        y_pred_c.append(classes[int(np.argmax(proba[idxs].mean(0)))])
    y_true_c = np.array(y_true_c); y_pred_c = np.array(y_pred_c)
    acc_c = accuracy_score(y_true_c, y_pred_c)
    f1_c = f1_score(y_true_c, y_pred_c, average="macro")
    print(f"  [{name}] clip-level: acc={acc_c:.4f} f1={f1_c:.4f}  ({len(y_true_c)} clips)")
    return {"frag_acc": acc, "frag_f1": f1, "clip_acc": acc_c, "clip_f1": f1_c,
            "clip_cm": confusion_matrix(y_true_c, y_pred_c, labels=classes).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--drop-tier1", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paper-acc", type=float, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[*] Loading {args.manifest}")
    X_ms, X_w4, X_py, y, groups = build_dataset(
        args.manifest, args.drop_tier1, args.seed)
    print(f"[+] N frags={len(y)}, N clips={len(set(groups))}")
    print(f"[+] dim: mean_std={X_ms.shape[1]}, windows_k4={X_w4.shape[1]}, pyramid={X_py.shape[1]}")

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_group = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    classes = np.array(sorted(np.unique(y)))

    results = {}

    # ---- baseline: pyramid alone, HGB tuned, fragment StratKF ----
    print("\n[1] Pyramid HGB alone, fragment-level StratKF")
    t0 = time.time()
    from sklearn.model_selection import cross_val_predict
    yp = cross_val_predict(hgb_tuned(args.seed), X_py, y, cv=cv_strat, n_jobs=4)
    acc = accuracy_score(y, yp); f1 = f1_score(y, yp, average="macro")
    results["pyramid_only_StratKF"] = {"acc": acc, "f1": f1,
                                       "delta": acc - args.paper_acc,
                                       "seconds": time.time() - t0}
    print(f"  acc={acc:.4f}  f1={f1:.4f}  Δ={acc-args.paper_acc:+.4f}  ({time.time()-t0:.1f}s)")

    # ---- pyramid alone under GroupKF + clip-level ----
    print("\n[2] Pyramid HGB alone, GroupKF + clip-level soft-vote")
    t0 = time.time()
    proba = np.zeros((len(y), len(classes)))
    for fi, (tr, te) in enumerate(cv_group.split(X_py, y, groups=groups)):
        m = hgb_tuned(args.seed); m.fit(X_py[tr], y[tr])
        proba[te] = m.predict_proba(X_py[te])
        print(f"     fold {fi+1}/5 done")
    res = evaluate("pyramid_only", y, proba, classes, groups)
    res.update({"delta_clip": res["clip_acc"] - args.paper_acc,
                "seconds": time.time() - t0})
    results["pyramid_only_GroupKF_clip"] = res

    # ---- stacking: mean_std + windows_k4 + pyramid -> meta-LogReg, GroupKF ----
    print("\n[3] Stacking [HGB(ms)+HGB(w4)+HGB(py)] -> meta-LR, GroupKF + clip-level")
    t0 = time.time()
    stacked_p, base_oof, _ = stacked_proba(
        [X_ms, X_w4, X_py], y, cv_group, groups=groups, seed=args.seed)
    res = evaluate("stacking", y, stacked_p, classes, groups)
    res.update({"delta_clip": res["clip_acc"] - args.paper_acc,
                "seconds": time.time() - t0})
    results["stacking_GroupKF_clip"] = res

    # ---- bonus: uniform average of the 3 base models (no meta), GroupKF + clip ----
    print("\n[4] Uniform soft-vote of 3 bases, GroupKF + clip-level")
    avg = sum(base_oof.values()) / len(base_oof)
    res = evaluate("uniform_avg3", y, avg, classes, groups)
    res.update({"delta_clip": res["clip_acc"] - args.paper_acc,
                "seconds": 0})
    results["uniform_avg3_GroupKF_clip"] = res

    # ---- print clip-level confusion matrix for best ----
    best_key = max(results, key=lambda k: results[k].get("clip_acc",
                                                          results[k].get("acc", 0)))
    print(f"\n[*] Best by clip_acc: {best_key}")
    cm = np.array(results[best_key].get("clip_cm", []))
    if cm.size:
        cm_n = cm / cm.sum(1, keepdims=True)
        print("    Per-clip confusion (rows=true, cols=pred):")
        print("         " + "   ".join(f"T{c}" for c in classes))
        for i, c in enumerate(classes):
            print(f"    T{c}:  " + "  ".join(f"{cm_n[i,j]*100:5.1f}"
                                              for j in range(len(classes))))

    # ---- save ----
    rows = []
    for k, v in results.items():
        rows.append({
            "experiment": k,
            "frag_acc": round(v.get("frag_acc", v.get("acc", float("nan"))), 4),
            "frag_f1":  round(v.get("frag_f1", v.get("f1", float("nan"))), 4),
            "clip_acc": round(v.get("clip_acc", float("nan")), 4),
            "clip_f1":  round(v.get("clip_f1", float("nan")), 4),
            "delta_clip_vs_paper": round(v.get("delta_clip",
                                                v.get("delta", float("nan"))), 4),
            "seconds": round(v.get("seconds", 0), 1),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[+] wrote {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
