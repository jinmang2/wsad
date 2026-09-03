#!/usr/bin/env bash
# Per-file download worker for a SharePoint/OneDrive folder.
# Args: <SourceUrl> <Name> <Length>
# Env : SP_COOKIE_FILE  SP_SITE  DEST_DIR
#
# Downloads one file via the _layouts/15/download.aspx?SourceUrl= endpoint,
# verifies byte length, retries up to 3x, and skips files already complete
# (makes the whole run resumable).
set -u
SRC="$1"; NAME="$2"; LEN="$3"
OUT="$DEST_DIR/$NAME"
COOKIE=$(cat "$SP_COOKIE_FILE")
DL="${SP_SITE}/_layouts/15/download.aspx"

# already complete?
if [ -f "$OUT" ]; then
  cur=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  [ "$cur" = "$LEN" ] && { echo "SKIP $NAME"; exit 0; }
fi

code="" cur=0
for try in 1 2 3; do
  code=$(curl -sS -L --max-time 1800 -G -b "$COOKIE" -H 'User-Agent: Mozilla/5.0' \
       --data-urlencode "SourceUrl=$SRC" "$DL" -o "$OUT" -w '%{http_code}')
  cur=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  if [ "$code" = "200" ] && [ "$cur" = "$LEN" ]; then
    echo "OK   $NAME ($cur)"; exit 0
  fi
  echo "RETRY($try) $NAME http=$code got=$cur want=$LEN"
  sleep 3
done
echo "FAIL $NAME http=$code got=$cur want=$LEN"
exit 1
