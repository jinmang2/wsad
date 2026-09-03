#!/usr/bin/env bash
# Download every file listed in a manifest TSV from a SharePoint/OneDrive folder.
# Resumable: files already present with the correct byte length are skipped.
#
# Usage: sp_download.sh <cookie_file> <site_base> <manifest_tsv> <dest_dir> [parallel]
#   parallel defaults to 4.
#
# On finish prints OK/SKIP/FAIL counts. Re-run with a fresh cookie to resume
# (e.g. after the FedAuth cookie expires mid-run) — completed files are skipped.
set -euo pipefail
export SP_COOKIE_FILE="$1"
export SP_SITE="$2"
MANIFEST="$3"
export DEST_DIR="$4"
P="${5:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DEST_DIR"

# Convert BOTH tab and newline to NUL so each (SourceUrl,Name,Length) field is a
# separate NUL token; xargs -0 -n3 then feeds exactly 3 args per file.
tr '\t\n' '\0\0' < "$MANIFEST" \
  | xargs -0 -n3 -P"$P" "$HERE/dl_one.sh"

echo "----"
echo "want : $(wc -l < "$MANIFEST")"
echo "have : $(ls -1 "$DEST_DIR" | wc -l)"
