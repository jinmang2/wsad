#!/usr/bin/env bash
# Per-crop FAITHFUL memory-head queue (the official training layout: each 10-crop file
# is a separate sample, 16100 rows). batch 64 (official) fits 8 GB because per-crop
# samples are n=1 (~10x lighter than the stacked (10,200,1024)); attention still needs
# WSAD_ATTN=mem (eager dots/repeat is (b,h,200,200)=8GB at b64). fp32, no checkpoint —
# the faithful config. Then the clean i3d 2048-d heads.
#
#   nohup setsid bash scripts/gpu_queue_pc.sh > outputs/gpu_queue.log 2>&1 < /dev/null & disown
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

PC="--variant i3d_1024_seg200_pc --feature-dim 1024 --batch 64 --eval-every 100 --eval-start 0"

# memory heads, per-crop, official batch 64, fp32 (faithful), mem attention for the b64 dots
run_job urdmu_pc_b64  mem --backbones i3d --heads ur_dmu  $PC --steps 3000 --force --out outputs/matrix_urdmu_pc_b64
run_job bnwvad_pc_b64 mem --backbones i3d --heads bn_wvad $PC --steps 1000 --force --out outputs/matrix_bnwvad_pc_b64

# clean i3d 2048-d heads + cosine (mgfn 0.8332 / sultani 0.8071 already done)
for h in s3r gs_moe; do
  run_job "${h}_clean_cosine" eager --backbones i3d --heads "$h" --variant i3d_mgfn_seg32 --lr-decay cosine --force --out "outputs/matrix_${h}_clean"
done

echo "===== [$(date +%H:%M:%S)] QUEUE DONE ====="
