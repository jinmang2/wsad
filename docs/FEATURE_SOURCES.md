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

## Screen result (2026-09-03): the best linear probe of any candidate, for zero GPU hours

Downloaded (33 min, 26.4 GB) and screened on the **same 40 videos** as every earlier gate.

| feature set | MAGNITUDE | CONTENT | PROBE | extraction cost |
|---|---|---|---|---|
| **`languagebind_10crop`** | 0.4972 | 0.6587 | **0.7144** | **zero GPU** |
| `videomae_GATEfix` | 0.5537 | **0.7209** | 0.6767 | ~3 h |
| `cosmos224_GATE` | 0.4932 | 0.6620 | 0.6272 | ~6 h |
| `i3d_1024_seg200` | 0.4995 | 0.5993 | 0.5536 | already have |
| `i3d_mgfn` | 0.4937 | 0.5916 | 0.5392 | already have |

LanguageBind takes the **linear probe** — the strongest of the three atemporal proxies — by
+0.038 over VideoMAE and +0.16 over the I3D features every reported number currently rests
on, while VideoMAE keeps the content-distance measure. Since the probe is the proxy that
best tracked downstream behaviour before, this is the most promising feature set the project
has seen, and it arrived without touching the GPU.

**Correction to the survey above.** These features are not L2-normalized (per-vector L2 ≈
12.6), and that was read as "magnitude-preserving, so RTFM/MGFN can use it". The screen says
otherwise: **magnitude AUC is 0.4972, i.e. chance.** Unnormalized is not the same as
informative. Magnitude-based heads should not be expected to gain anything here; the value
is in the content/probe axes, which is where the VLM and attention heads live.

Still only a *ranking*. The real gate is a short head train (`run_matrix.py --variant
languagebind_10crop --feature-dim 768`), which needs the GPU and has not been run.

### Integration fixes this required (all of them silent failures)

- **Test-cache layout.** `languagebind_10crop` stores `(ncrops, T, D)` in *both* splits,
  whereas the I3D test cache is `(T, ncrops, D)` — the same video is `(10, 88, 768)` here and
  `(88, 10, 1024)` there. Everything downstream assumes the I3D convention for i3d-backbone
  variants, so the features are transposed at load (`_CROPS_FIRST_TEST_VARIANTS`). Without
  it, 88 snippets are read as 10 crops and the ground truth silently misaligns.
- **Video-id anchoring.** `_bare_vid` stripped a literal `_i3d` suffix, so
  `Abuse028_x264_languagebind.npy` matched no ground-truth key and *every* test video was
  dropped. Now anchored on `_x264`, which every UCF-Crime id ends with, so any future
  backbone tag works without an edit.
- **Diagnostic layout.** `feature_forensics.py` assumed `(T, ncrops, D)` too; `--crops-first`
  transposes. Without it the crop mean is taken over time and the numbers are meaningless
  rather than absent.
