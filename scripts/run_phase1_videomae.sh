#!/usr/bin/env bash
# Robust, RESUMABLE Phase-1 runner: VideoMAE features -> content heads, vs I3D.
# The WSL host reboots intermittently (kills background jobs + /tmp scripts), so this
# lives in the repo and every stage is IDEMPOTENT — just re-run it after any reboot and
# it skips finished work and continues. Extraction skips existing .npy; head training
# skips a stage whose results.json already exists.
#
#   nohup setsid bash scripts/run_phase1_videomae.sh > outputs/phase1_videomae.log 2>&1 < /dev/null & disown
#
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. WSAD_DATA="${WSAD_DATA:-$HOME/data/wsad}" PYTHONUNBUFFERED=1
RUN="conda run --no-capture-output -n balaenoptera python -u"
FEAT="$WSAD_DATA/ucf_crime/features/videomae_seg32"
MODEL="MCG-NJU/videomae-base"
TRAIN_N=1610; TEST_N=290

ts() { date +%H:%M:%S; }

# ---- stage 1: extract train (sample_to 32, fast ~1.3s/vid). resumable. ----
have=$(ls "$FEAT/train" 2>/dev/null | wc -l)
if [ "$have" -lt "$TRAIN_N" ]; then
  echo "[$(ts)] EXTRACT train (have $have/$TRAIN_N)"
  $RUN scripts/extract_modern.py --backbone videomae --model-name $MODEL --split train \
    --sample-to 32 --device cuda --out "$FEAT/train" >> outputs/p1_ex_train.log 2>&1
fi
echo "[$(ts)] train ready: $(ls "$FEAT/train" 2>/dev/null | wc -l)/$TRAIN_N"

# ---- stage 2: extract test (full-length, needed for frame-eval). resumable. ----
have=$(ls "$FEAT/test" 2>/dev/null | wc -l)
if [ "$have" -lt "$TEST_N" ]; then
  echo "[$(ts)] EXTRACT test (have $have/$TEST_N)"
  $RUN scripts/extract_modern.py --backbone videomae --model-name $MODEL --split test \
    --device cuda --out "$FEAT/test" >> outputs/p1_ex_test.log 2>&1
fi
echo "[$(ts)] test ready: $(ls "$FEAT/test" 2>/dev/null | wc -l)/$TEST_N"

# only train heads once BOTH splits are complete
if [ "$(ls "$FEAT/train" 2>/dev/null | wc -l)" -lt "$TRAIN_N" ] || \
   [ "$(ls "$FEAT/test" 2>/dev/null | wc -l)" -lt "$TEST_N" ]; then
  echo "[$(ts)] extraction incomplete — re-run this script after the next reboot to resume."
  exit 0
fi

# ---- stage 3: content-head training (skip if result exists). ----
train_head () {  # head  extra-args...
  local h="$1"; shift
  local out="outputs/matrix_${h}_videomae"
  if [ -f "$out/results.json" ] && ! grep -q '"error"' "$out/results.json"; then
    echo "[$(ts)] $h @ videomae already done: $($RUN -c "import json;print(json.load(open('$out/results.json'))[0].get('roc_auc'))" 2>/dev/null)"
    return
  fi
  echo "[$(ts)] TRAIN $h @ videomae"
  $RUN scripts/run_matrix.py --backbones i3d --heads "$h" --variant videomae_seg32 \
    --feature-dim 768 --lr-decay cosine "$@" --force --out "$out" > "outputs/p1_${h}.log" 2>&1
  $RUN -c "import json;r=json.load(open('$out/results.json'))[0];print('[$(ts)] $h @ videomae roc=%s'%r.get('roc_auc'))" 2>/dev/null
}
train_head sultani --eval-every 200
train_head ur_dmu --batch 64 --eval-every 100 --eval-start 0 --steps 3000

echo "[$(ts)] PHASE1 DONE — sultani vs i3d 0.807, ur_dmu vs i3d 0.815"
