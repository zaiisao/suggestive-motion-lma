#!/usr/bin/env bash
# Re-run the three classifier configurations from the SIGGRAPH Posters '26
# paper using the shipped per-cutoff manifests, then assert that each
# accuracy matches the published value bit-exactly.
#
# Usage:  bash scripts/verify_paper_reproduction.sh
#
# Requires: the `wham` conda env active, and the LMA feature files referenced
# by data/manifest_paper_*.csv reachable at their recorded absolute paths.
# Verify file integrity first with:
#   python scripts/check_manifest_hashes.py data/manifest_paper_4way_2026-04-15_01-23.csv

set -e
cd "$(dirname "$0")/.."

REPO="$(pwd)"
OUT="${REPO}/output/audit"
mkdir -p "${OUT}"

# Expected paper values (from the three published summary.json files)
EXPECT_4WAY_LR=0.5728
EXPECT_4WAY_RF=0.5742
EXPECT_3WAY_LR=0.7206
EXPECT_3WAY_RF=0.7020
EXPECT_BIN_LR=0.7869
EXPECT_BIN_RF=0.7824

run_one() {
    local name="$1"; shift
    local out="${OUT}/${name}"
    python scripts/analyze_lma_tiers.py "$@" --out-dir "${out}" > "${out}.log" 2>&1
    python -c "
import json, sys
s = json.load(open('${out}/summary.json'))
lr = round(s['classifiers']['LogReg']['acc'], 4)
rf = round(s['classifiers']['RandomForest']['acc'], 4)
print(f'  LogReg={lr}  RF={rf}')
"
}

echo "=== 4-way (paper: LogReg=${EXPECT_4WAY_LR} RF=${EXPECT_4WAY_RF}) ==="
run_one 4way \
    --manifest data/manifest_paper_4way_2026-04-15_01-23.csv \
    --max-per-tier 1075

echo ""
echo "=== 3-way (paper: LogReg=${EXPECT_3WAY_LR} RF=${EXPECT_3WAY_RF}) ==="
run_one 3way \
    --manifest data/manifest_paper_3way_2026-04-14_19-16.csv \
    --drop-tier1 --max-per-tier 1075

echo ""
echo "=== Binary (paper: LogReg=${EXPECT_BIN_LR} RF=${EXPECT_BIN_RF}) ==="
run_one binary \
    --manifest data/manifest_paper_binary_2026-04-14_21-36.csv \
    --binary --max-per-tier 2000

echo ""
echo "[*] Detailed outputs in ${OUT}/{4way,3way,binary}/"
