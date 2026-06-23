#!/bin/bash
export SP_SITE='https://stuxidianeducn-my.sharepoint.com/personal/pengwu_stu_xidian_edu_cn'
export SP_COOKIE_FILE=/tmp/sp_cookie.txt
export DEST_DIR="$HOME/data/wsad/ucf_crime/features/i3d_1024_raw/test_npy"
cd "$HOME/wsad" || exit 1
export -f 2>/dev/null
# parallel per-file (small files -> overhead-bound, 8-way)
cat /tmp/test_manifest.tsv | xargs -P 8 -d '\n' -I {} bash -c '
  IFS=$'"'"'\t'"'"' read -r src name len <<< "{}"
  SP_SITE="'"$SP_SITE"'" SP_COOKIE_FILE=/tmp/sp_cookie.txt DEST_DIR="'"$DEST_DIR"'" bash scripts/onedrive_fetch/dl_one.sh "$src" "$name" "$len" >/dev/null 2>&1
'
echo "=== TEST DOWNLOAD DONE $(date +%H:%M:%S); files: $(ls $DEST_DIR/*.npy 2>/dev/null | wc -l)/2901 ==="
