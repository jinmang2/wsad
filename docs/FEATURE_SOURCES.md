# Pre-extracted UCF-Crime features — survey (2026-09-02)

Motivation: extraction is the binding constraint on this project, not training. The single
8 GB RTX 2070 SUPER needs ~3 h for VideoMAE-base and ~6–24 h for Cosmos-Embed1 to cover the
1900-video set (`docs/NEXT_PHASE.md`). **A published feature set costs zero GPU hours**, so
it is worth a hard look before spending another night extracting.

Surveyed the Hugging Face dataset index for UCF-Crime / XD-Violence feature dumps and
inspected the actual arrays (shape, dim, whether the vectors are L2-normalized).

## The find: LanguageBind, full coverage, drop-in layout

`yukaneko55/UCF-Crime_features` — `Languagebind/<stem>_languagebind.npy`, **1900 files**,
26.4 GB, mean 13.9 MB.

| property | value | why it matters |
|---|---|---|
| shape | `(10, T, 768)` | already the `(ncrops, T, D)` layout `_to_crops_layout` wants — 10-crop, no reshaping |
| snippet grid | T=170 for `Abuse001` (171 I3D snippets) | same 16-frame grid as I3D/VideoMAE — seg32/seg200 conventions carry over |
| L2 norm | mean 12.59, **not normalized** | magnitude survives, so RTFM/MGFN work (unlike Cosmos `proj` and MobileCLIP2) |
| coverage | 950 anomaly + 950 normal = exactly our 1610 train + 290 test | complete, no gaps to patch |
| filename | `<stem>_languagebind.npy` | already our `<stem>_<backbone>.npy` cache convention |
| text-aligned | yes — LanguageBind's language tower is a frozen 768-d OpenCLIP transformer | unlocks the VLM heads too, not just the visual ones |
| leakage | **clean** — trained on VIDAL-10M (video/infrared/depth/audio + language, short-video sourced), no UCF-Crime | safe to evaluate on our test split |

So a single modern, text-aligned, magnitude-preserving backbone becomes available for
**zero GPU time**, and it is the only candidate found that serves the magnitude heads and
the VLM heads at once.

⚠️ **Provenance is unverified.** The repo was uploaded 2026-08-06 with no model card, no
README and no extraction script, by an account with no other relevant work. The arrays are
shaped and scaled plausibly, but nothing documents which LanguageBind checkpoint, which
crop protocol, or which stride produced them. Treat exactly like any other candidate: run
`scripts/diag/feature_forensics.py` first, then a short `run_matrix.py` head train, and
record the outcome in `docs/RESULTS_TABLE.md` with the provenance caveat attached. Do not
report a number from these features as a clean backbone comparison until the screen and the
head train both behave.

## Other candidates inspected

| dataset | content | verdict |
|---|---|---|
| `jisujang/UCF_Crimes_MobileCLIP2-S0-F6th` | MobileCLIP2-S0, `(T, 512)` single-crop, **L2-normalized**, T=90 for `Abuse001` (a different, sparser grid); also ships per-class timestamp annotations | Interesting for Spec 2 — MobileCLIP2 is built for real-time, and 512-d matches VadCLIP's CLIP-512 text tower exactly. But normalized (no magnitude) and off-grid, so it needs its own loader path. Second priority. |
| `Kavindu1124/ucf-crime-processed-features` | `(T, 2048)` single-crop `.npz` | Almost certainly I3D, which we already have in two variants. Skip. |
| `myzhao1999/ucf-crime-clip-features`, `Ship2001/CLIP_for_...` | CLIP dumps in one zip | We extract CLIP ourselves already. Skip. |
| `backseollgi/lavad_ucf_crime_features` | LAVAD vision features (one zip) | LAVAD is an LLM-based VAD pipeline; its features are pipeline-specific rather than a general backbone. Low priority. |
| `SabrianLinnn/UCF-Crime_10-crop_I3D_features` | 10-crop I3D | Duplicate of `i3d_mgfn`. Skip. |
| `KingTechnician/xd-violence-rgb-videomae-chunked-*` | VideoMAE features for **XD-Violence** | Wrong dataset for the current table, but it is the ready-made route if XD-Violence (the listed secondary benchmark) is ever taken up — VideoMAE there costs no GPU either. |

## Recommended order

1. **LanguageBind first.** 26 GB of download instead of hours of GPU, it is the only
   candidate that serves every head class, and the forensic screen that validates it is
   minutes of CPU. `scripts/fetch_pretrained_features.py` is ready to fetch and arrange it
   into `features/languagebind_10crop/{train,test}`.
2. Screen it against `i3d_mgfn` / `i3d_1024` / `videomae_GATEfix` on the same 40 videos,
   then a short `sultani` + `ur_dmu` train at `--feature-dim 768`.
3. Only if it fails the screen does the Cosmos/VideoMAE extraction budget become the best
   use of the card again.
