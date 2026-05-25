"""
LMA tier analysis — the classifier from "Appearance-Invariant Detection of
Suggestive Motion via Laban Movement Descriptors on SMPL Skeletons"
(SIGGRAPH Posters '26).

Loads pre-computed LMA fragments from tiers 0, 1, 2, 3 and runs:
  1. Per-tier feature statistics
  2. ANOVA / Kruskal-Wallis per-feature discriminability
  3. PCA 2D visualization
  4. t-SNE 2D visualization
  5. Logistic regression + RandomForest classifiers (5-fold CV)
  6. Feature importance

Outputs plots + CSVs to --out-dir.

Tier directories are resolved (in order) from:
  1. --tierN-dirs flag (one or more paths)
  2. LMA_TIER{N}_DIRS environment variable (os.pathsep-separated)
  3. ./data/tierN/ (default)

Each directory is searched recursively for files matching `lma_features_*.npy`,
each of which should hold a (T, 55) matrix of frame-wise LMA descriptors.

Usage:
  python scripts/analyze_lma_tiers.py                     # 4-way
  python scripts/analyze_lma_tiers.py --drop-tier1        # 3-way
  python scripts/analyze_lma_tiers.py --binary            # SFW/NSFW
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy import stats


def _resolve_tier_dirs(tier_idx, cli_dirs):
    if cli_dirs:
        return list(cli_dirs)
    env = os.environ.get(f"LMA_TIER{tier_idx}_DIRS")
    if env:
        return [p for p in env.split(os.pathsep) if p]
    return [os.path.join("data", f"tier{tier_idx}")]


def _files_from_manifest(manifest_path):
    """Return {tier: [paths]} from a manifest CSV produced by build_manifest.py."""
    out = {0: [], 1: [], 2: [], 3: []}
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["tier"])
            out[t].append(row["path"])
    for t in out:
        out[t].sort()
    return out


def _files_from_dirs(roots):
    files = []
    for root in roots:
        files.extend(glob.glob(os.path.join(root, "**/lma_features_*.npy"), recursive=True))
    return sorted(set(files))

# Same feature order as sorted keys in lma_dict
FEATURE_NAMES = [
    "Dispersion_Head", "Dispersion_L_Ankle", "Dispersion_L_Wrist",
    "Dispersion_R_Ankle", "Dispersion_R_Wrist",
    "Dist_Ankle_Knee_L", "Dist_Ankle_Knee_R", "Dist_Feet",
    "Dist_Hand_Shoulder_L", "Dist_Hand_Shoulder_R", "Dist_Hands",
    "Effort_Flow_Global", "Effort_Space_Global", "Effort_Time_Global",
    "Effort_Weight_Global",
    "HEAD_Accel", "HEAD_Directness", "HEAD_Jerk", "HEAD_KE", "HEAD_vel",
    "Initiation_HEAD", "Initiation_L_ANKLE", "Initiation_L_WRIST",
    "Initiation_PELVIS", "Initiation_R_ANKLE", "Initiation_R_WRIST",
    "L_ANKLE_Accel", "L_ANKLE_Directness", "L_ANKLE_Jerk",
    "L_ANKLE_KE", "L_ANKLE_vel",
    "L_WRIST_Accel", "L_WRIST_Directness", "L_WRIST_Jerk",
    "L_WRIST_KE", "L_WRIST_vel",
    "PELVIS_Accel", "PELVIS_Directness", "PELVIS_Jerk",
    "PELVIS_KE", "PELVIS_vel",
    "R_ANKLE_Accel", "R_ANKLE_Directness", "R_ANKLE_Jerk",
    "R_ANKLE_KE", "R_ANKLE_vel",
    "R_WRIST_Accel", "R_WRIST_Directness", "R_WRIST_Jerk",
    "R_WRIST_KE", "R_WRIST_vel",
    "Traj_Curvature", "Traj_Displacement", "Traj_Path_Length", "body_volume",
]

TIER_COLORS = {0: "#4C72B0", 1: "#55A868", 2: "#C44E52", 3: "#8172B2"}
TIER_NAMES = {0: "Tier 0 (Normal)", 1: "Tier 1 (Artistic)",
              2: "Tier 2 (Suggestive)", 3: "Tier 3 (Explicit)"}


def aggregate_fragment(arr):
    """Collapse (T, 55) into per-fragment summary.
    Returns mean(T) + std(T) concatenated -> (110,) vector.
    """
    if arr.ndim != 2 or arr.shape[1] != 55:
        return None
    if arr.shape[0] < 5:  # too short to be meaningful
        return None
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    # Replace nan/inf
    out = np.concatenate([mu, sigma])
    if not np.all(np.isfinite(out)):
        return None
    return out


def load_files(files):
    """Load fragments from an explicit file list, return (N, 110) matrix."""
    rows = []
    for f in files:
        try:
            arr = np.load(f, allow_pickle=True)
            vec = aggregate_fragment(arr)
            if vec is not None:
                rows.append(vec)
        except Exception:
            pass
    if not rows:
        return np.zeros((0, 110))
    return np.stack(rows, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-tier1-into-tier0", action="store_true",
                        help="Merge tier 1 fragments into tier 0")
    parser.add_argument("--binary", action="store_true",
                        help="Binary classification: {0,1} vs {2,3}")
    parser.add_argument("--drop-tier1", action="store_true",
                        help="Exclude tier 1 entirely")
    parser.add_argument("--out-dir", default="output/lma_analysis")
    parser.add_argument("--max-per-tier", type=int, default=2000,
                        help="Cap fragments per tier to keep classes balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default=None,
                        help="CSV manifest of files to use (overrides --tierN-dirs). "
                             "Format: tier,path,mtime_utc,size_bytes,sha256. "
                             "Produced by scripts/build_manifest.py.")
    for t in (0, 1, 2, 3):
        parser.add_argument(f"--tier{t}-dirs", nargs="+", default=None,
                            help=f"Directories of pre-computed Tier {t} LMA features "
                                 f"(overrides $LMA_TIER{t}_DIRS).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    # --- 1. Load data ---
    print("[*] Loading features...")
    tier_data = {}
    if args.manifest:
        print(f"[*] Manifest mode: {args.manifest}")
        tier_files = _files_from_manifest(args.manifest)
    else:
        tier_files = {
            t: _files_from_dirs(_resolve_tier_dirs(t, getattr(args, f"tier{t}_dirs")))
            for t in (0, 1, 2, 3)
        }
    for tier in (0, 1, 2, 3):
        files = tier_files[tier]
        print(f"  Tier {tier}: {len(files)} files listed")
        X = load_files(files)
        print(f"    -> {X.shape[0]} valid fragments")
        tier_data[tier] = X

    # Apply tier merging / dropping
    if args.merge_tier1_into_tier0:
        tier_data[0] = np.concatenate([tier_data[0], tier_data[1]], axis=0)
        del tier_data[1]
        print("[+] Merged Tier 1 into Tier 0")
    elif args.drop_tier1:
        del tier_data[1]
        print("[+] Dropped Tier 1")

    # Binary: {0,1} vs {2,3}
    if args.binary:
        sfw = np.concatenate([tier_data.get(0, np.zeros((0,110))),
                              tier_data.get(1, np.zeros((0,110)))], axis=0)
        nsfw = np.concatenate([tier_data.get(2, np.zeros((0,110))),
                               tier_data.get(3, np.zeros((0,110)))], axis=0)
        tier_data = {0: sfw, 1: nsfw}
        label_names = {0: "SFW (Tier 0+1)", 1: "NSFW (Tier 2+3)"}
    else:
        label_names = {t: TIER_NAMES[t] for t in tier_data}

    # Balance
    min_n = min(X.shape[0] for X in tier_data.values())
    max_n = min(args.max_per_tier, min_n)
    print(f"[*] Balancing to {max_n} fragments per class")
    balanced = {}
    for k, X in tier_data.items():
        idx = np.random.choice(X.shape[0], size=max_n, replace=False)
        balanced[k] = X[idx]

    # Assemble X, y
    X_list, y_list = [], []
    for k, Xk in balanced.items():
        X_list.append(Xk)
        y_list.append(np.full(Xk.shape[0], k))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"[+] Final dataset: X={X.shape}, y={y.shape}")

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Handle any remaining NaN
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 2. Per-tier feature statistics table ---
    print("\n[*] Computing per-tier feature statistics...")
    feat_names_full = [f"{n}_mean" for n in FEATURE_NAMES] + [f"{n}_std" for n in FEATURE_NAMES]
    stats_rows = []
    for k in sorted(balanced.keys()):
        Xk = balanced[k]
        row = {"label": label_names[k], "n": Xk.shape[0]}
        for i, fn in enumerate(feat_names_full):
            row[f"{fn}_mean"] = Xk[:, i].mean()
        stats_rows.append(row)
    pd.DataFrame(stats_rows).to_csv(
        os.path.join(args.out_dir, "per_tier_means.csv"), index=False)

    # --- 3. Discriminative features: Kruskal-Wallis H test ---
    print("[*] Ranking features by Kruskal-Wallis H...")
    groups = {k: Xs[y == k] for k in np.unique(y)}
    kw_rows = []
    for i, fn in enumerate(feat_names_full):
        samples = [g[:, i] for g in groups.values()]
        try:
            H, p = stats.kruskal(*samples)
        except Exception:
            H, p = 0.0, 1.0
        kw_rows.append({"feature": fn, "H": H, "p_value": p})
    kw_df = pd.DataFrame(kw_rows).sort_values("H", ascending=False)
    kw_df.to_csv(os.path.join(args.out_dir, "kruskal_features.csv"), index=False)
    print("\n  Top 15 discriminative features (by Kruskal-Wallis H):")
    for _, row in kw_df.head(15).iterrows():
        print(f"    {row['feature']:40s}  H={row['H']:10.2f}  p={row['p_value']:.2e}")

    # --- 4. PCA 2D plot ---
    print("\n[*] Computing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xs)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    for k in sorted(balanced.keys()):
        mask = (y == k)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=TIER_COLORS.get(k, "#888"), label=label_names[k],
                   alpha=0.5, s=15)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("LMA Features — PCA 2D projection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pca_2d.png"), dpi=150)
    plt.close()

    # --- 5. t-SNE plot ---
    print("[*] Computing t-SNE (may take a minute)...")
    # Subsample for speed
    if X.shape[0] > 2000:
        tsne_idx = np.random.choice(X.shape[0], 2000, replace=False)
    else:
        tsne_idx = np.arange(X.shape[0])
    tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed, init="pca")
    X_tsne = tsne.fit_transform(Xs[tsne_idx])
    y_tsne = y[tsne_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    for k in sorted(balanced.keys()):
        mask = (y_tsne == k)
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=TIER_COLORS.get(k, "#888"), label=label_names[k],
                   alpha=0.5, s=15)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("LMA Features — t-SNE 2D projection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "tsne_2d.png"), dpi=150)
    plt.close()

    # --- 6. Classifiers (5-fold CV) ---
    print("\n[*] Training classifiers (5-fold CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    results = {}
    for name, model in [
        ("LogReg", LogisticRegression(max_iter=2000)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)),
    ]:
        y_pred = cross_val_predict(model, Xs, y, cv=cv, n_jobs=-1)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        cm = confusion_matrix(y, y_pred)
        results[name] = {"acc": acc, "f1_macro": f1, "cm": cm.tolist()}
        print(f"  {name:15s}  acc={acc:.4f}  f1_macro={f1:.4f}")

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(7, 6))
        labels_sorted = sorted(balanced.keys())
        label_txt = [label_names[k] for k in labels_sorted]
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels_sorted)))
        ax.set_yticks(range(len(labels_sorted)))
        ax.set_xticklabels(label_txt, rotation=30, ha="right")
        ax.set_yticklabels(label_txt)
        for i in range(len(labels_sorted)):
            for j in range(len(labels_sorted)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_title(f"{name} confusion (row-normalized)\nacc={acc:.3f}  f1_macro={f1:.3f}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"cm_{name.lower()}.png"), dpi=150)
        plt.close()

    with open(os.path.join(args.out_dir, "classifier_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- 7. Feature importance from RandomForest (fit on full data) ---
    print("\n[*] Computing RF feature importance...")
    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1)
    rf.fit(Xs, y)
    imp = pd.DataFrame({"feature": feat_names_full, "importance": rf.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(args.out_dir, "rf_feature_importance.csv"), index=False)
    print("\n  Top 15 features (RF importance):")
    for _, row in imp.head(15).iterrows():
        print(f"    {row['feature']:40s}  {row['importance']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 10))
    top = imp.head(20).iloc[::-1]
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 LMA features (RandomForest importance)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "rf_importance.png"), dpi=150)
    plt.close()

    # --- 8. Summary JSON ---
    summary = {
        "n_per_class": {str(k): int(balanced[k].shape[0]) for k in sorted(balanced.keys())},
        "labels": {str(k): v for k, v in label_names.items()},
        "pca_explained": float(pca.explained_variance_ratio_.sum()),
        "classifiers": results,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
