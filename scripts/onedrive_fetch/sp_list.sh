#!/usr/bin/env bash
# List every file in a SharePoint/OneDrive folder via REST -> manifest TSV.
# Usage: sp_list.sh <cookie_file> <site_base> <folder_server_relative_url> <out_tsv>
#
#   cookie_file : text file holding the raw Cookie header value (FedAuth=...; ...)
#   site_base   : https://<tenant>-my.sharepoint.com/personal/<user>
#   folder      : server-relative URL, e.g.
#                 /personal/<user>/Documents/UCF-Crime/UCF_Train_ten_i3d
#   out_tsv     : output manifest, 3 cols: ServerRelativeUrl <tab> Name <tab> Length
#
# NOTE: $top=5000 returns up to 5000 files in one page; folders larger than that
#       need paging (not implemented — none of the UCF folders need it).
set -euo pipefail
COOKIE_FILE="$1"; SITE="$2"; FOLDER="$3"; OUT="$4"
COOKIE=$(cat "$COOKIE_FILE")
TMP=$(mktemp)
curl -fsS --max-time 180 -b "$COOKIE" \
  -H 'Accept: application/json;odata=nometadata' \
  "${SITE}/_api/web/GetFolderByServerRelativeUrl('${FOLDER}')/Files?\$select=Name,ServerRelativeUrl,Length&\$top=5000" \
  -o "$TMP"
python3 - "$TMP" "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); v = d['value']
with open(sys.argv[2], 'w') as f:
    for x in v:
        f.write(f"{x['ServerRelativeUrl']}\t{x['Name']}\t{x['Length']}\n")
gb = sum(int(x['Length']) for x in v) / 1e9
print(f"{len(v)} files, {gb:.2f} GB -> {sys.argv[2]}")
PY
rm -f "$TMP"
