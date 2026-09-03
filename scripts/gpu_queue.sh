#!/usr/bin/env bash
# Serial GPU queue (8 GB -> ONE heavy run at a time). MEMORY HEADS FIRST (the open
# 0.80->0.87 question), then the clean i3d 2048-d heads. Memory heads use the full
# 8 GB toolkit (mem attention + gradient checkpointing + fp16) at batch 32 (the 8 GB
# ceiling; b64 OOMs) PLUS the two remaining 8 GB levers: cosine LR decay (the rtfm
# 0.81->0.834 lever) and EXTENDED iters (bn_wvad's peak was still rising at 1000).
#
#   nohup setsid bash scripts/gpu_queue.sh > outputs/gpu_queue.log 2>&1 < /dev/null & disown
#
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export WSAD_DATA="${WSAD_DATA:-$HOME/data/wsad}"
export PYTHONUNBUFFERED=1
RUN="uv run python -u scripts/run_matrix.py"

run_job () {  # tag  ATTN  <run_matrix args...>
  local tag="$1"; local attn="$2"; shift 2
  echo "===== [$(date +%H:%M:%S)] START $tag ====="
  WSAD_ATTN="$attn" $RUN "$@" > "outputs/${tag}.log" 2>&1
  echo "===== [$(date +%H:%M:%S)] END   $tag (exit $?) ====="
  tail -3 "outputs/${tag}.log"
}

MEM="--batch 32 --grad-checkpoint --mixed-precision fp16 --lr-decay cosine --eval-every 50 --eval-start 0"

# --- Track 2: memory heads, batch 32 full-toolkit + cosine + extended iters ---
run_job urdmu_1024_b32cos  mem --backbones i3d --heads ur_dmu  --variant i3d_1024_seg200 --feature-dim 1024 $MEM --steps 3000 --force --out outputs/matrix_urdmu_1024_b32cos
run_job bnwvad_1024_b32cos mem --backbones i3d --heads bn_wvad --variant i3d_1024_seg200 --feature-dim 1024 $MEM --steps 3000 --force --out outputs/matrix_bnwvad_1024_b32cos

# --- Track 1: clean i3d 2048-d heads + cosine (mgfn already 0.8332, sultani 0.8071 done) ---
for h in s3r gs_moe; do
  run_job "${h}_clean_cosine" eager --backbones i3d --heads "$h" --variant i3d_mgfn_seg32 --lr-decay cosine --force --out "outputs/matrix_${h}_clean"
done

echo "===== [$(date +%H:%M:%S)] QUEUE DONE ====="
