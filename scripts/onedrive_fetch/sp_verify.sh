#!/usr/bin/env bash
# Verify a downloaded folder against its manifest TSV.
# Usage: sp_verify.sh <manifest_tsv> <dest_dir>
# Reports missing files, wrong-size files, and confirms NumPy magic on a sample.
set -euo pipefail
MANIFEST="$1"; DEST="$2"
python3 - "$MANIFEST" "$DEST" <<'PY'
import os, sys
manifest, dest = sys.argv[1], sys.argv[2]
want = {}
for line in open(manifest):
    src, name, length = line.rstrip('\n').split('\t')
    want[name] = int(length)
missing, wrong = [], []
for name, length in want.items():
    p = os.path.join(dest, name)
    if not os.path.exists(p):
        missing.append(name)
    elif os.path.getsize(p) != length:
        wrong.append((name, os.path.getsize(p), length))
print(f"manifest : {len(want)} files")
print(f"on disk  : {sum(1 for n in want if os.path.exists(os.path.join(dest,n)))}")
print(f"missing  : {len(missing)}")
print(f"badsize  : {len(wrong)}")
for n in missing[:20]: print("  MISSING", n)
for n,g,w in wrong[:20]: print(f"  BADSIZE {n} got={g} want={w}")
# sample NumPy magic check
ok = True
for n in list(want)[:5]:
    p = os.path.join(dest, n)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            if f.read(6) != b'\x93NUMPY':
                ok = False; print("  NOT-NPY", n)
print("sample npy magic:", "OK" if ok else "FAIL")
sys.exit(1 if (missing or wrong) else 0)
PY
