# onedrive_fetch — download a SharePoint/OneDrive folder file-by-file

Built to recover the **MGFN pre-seg32 ten-crop UCF-Crime I3D features** from the HKU
OneDrive share (`connecthkuhk-my.sharepoint.com/.../UCF-Crime`).

## Why this exists
The data is a **folder of individual `.npy` files** (Train = 1610 files / 64.87 GB,
Test = 290 files). Clicking **Download** on the whole folder makes OneDrive zip it
on the fly, which hits SharePoint's size limit and fails:

> The file size exceeds the allowed limit. CorrelationId: ...

The share is an **anonymous link** (no interactive login possible), but the
`_layouts/15/download.aspx?SourceUrl=<path>` endpoint works **per-file** as long as
you send the browser's session **cookie**. So we list the folder via REST, then pull
each file individually. No zip, no size limit.

## One thing only you can do: grab the cookie
1. Open the share folder in the browser (you're already authenticated there).
2. F12 → **Network** tab → filter **Fetch/XHR** → refresh (F5).
3. Right-click a `RenderListDataAsStream` (or any `connecthkuhk-my.sharepoint.com`)
   request with status **200** → **Copy → Copy as cURL (bash)**.
4. From the pasted command, take the value after `-b '...'` (or `-H 'cookie: ...'`)
   — the `FedAuth=...; ...` string — and save it as a single line:
   ```bash
   printf '%s' 'FedAuth=...; rtFa=...; ...' > /tmp/sp_cookie.txt
   chmod 600 /tmp/sp_cookie.txt
   ```
   FedAuth is short-lived (~1 h). If a run starts failing with http=403, re-copy a
   fresh cookie and re-run — completed files are skipped.

## Run
```bash
SITE='https://connecthkuhk-my.sharepoint.com/personal/cyxcarol_connect_hku_hk'
COOKIE=/tmp/sp_cookie.txt

# 1) list the folder -> manifest TSV (ServerRelativeUrl \t Name \t Length)
scripts/onedrive_fetch/sp_list.sh "$COOKIE" "$SITE" \
  '/personal/cyxcarol_connect_hku_hk/Documents/UCF-Crime/UCF_Train_ten_i3d' \
  /tmp/train_manifest.tsv

# 2) download all files (parallel 4, resumable, size-verified)
scripts/onedrive_fetch/sp_download.sh "$COOKIE" "$SITE" \
  /tmp/train_manifest.tsv ~/data/UCF_Train_ten_i3d 4

# 3) verify count + that every file matches its manifest length
scripts/onedrive_fetch/sp_verify.sh /tmp/train_manifest.tsv ~/data/UCF_Train_ten_i3d
```
Test set: same commands with `UCF_Test_ten_i3d` and a `test_manifest.tsv`.

## Files
- `sp_list.sh`     — REST folder listing → manifest TSV (uses `GetFolderByServerRelativeUrl/Files`, `$top=5000`).
- `sp_download.sh` — orchestrator: manifest → parallel resumable download.
- `dl_one.sh`      — per-file worker (download.aspx?SourceUrl, size-check, 3 retries).
- `sp_verify.sh`   — checks on-disk files against the manifest (count + per-file length).

## Notes
- Resumability is by **exact byte length** match vs the manifest, so a truncated file
  is re-fetched on the next run.
- `$top=5000` is a single page; folders >5000 files would need paging (none here).
- The captured cookie authorises whoever holds it for the share's lifetime — keep
  `/tmp/sp_cookie.txt` local (mode 600) and let it expire; do not commit it.
