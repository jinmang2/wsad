#!/bin/bash
# one-off: sequentially download the 3 DeepMIL UCF 1024-d I3D train zips.
# sequential = each gets full bandwidth -> finishes within dl_one's 30-min max-time.
export SP_SITE='https://stuxidianeducn-my.sharepoint.com/personal/pengwu_stu_xidian_edu_cn'
export SP_COOKIE_FILE=/tmp/sp_cookie.txt
export DEST_DIR="$HOME/data/wsad/ucf_crime/features/i3d_1024_raw/train"
cd "$HOME/wsad" || exit 1
mkdir -p "$DEST_DIR"
while IFS=$'\t' read -r src name len; do
  echo "=== $(date +%H:%M:%S) start $name ($len) ==="
  bash scripts/onedrive_fetch/dl_one.sh "$src" "$name" "$len"
  echo "=== $(date +%H:%M:%S) done $name ==="
done < /tmp/train_zips.tsv
echo "=== ALL TRAIN ZIPS DONE $(date +%H:%M:%S) ==="
