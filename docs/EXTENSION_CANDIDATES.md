# Extension candidates — heads, features, and prior art (survey 2026-09-02)

Research pass done while the GPU was fully occupied by the Cosmos gate (88% util, 207/215 W),
so everything here is desk work: what is worth adding to the comparison matrix, what it
would cost, and what it competes with. Nothing below has been run.

Current matrix (`docs/RESULTS_TABLE.md`): 10 heads on I3D-2048 / I3D-1024 / CLIP, best
reproduction 0.8654 (VadCLIP), paper-best 0.9160 (GS-MoE).

## 1. Heads worth adding (all feature-cached, so they drop into `run_matrix.py`)

| head | UCF AUC | features | code | why it is interesting |
|---|---|---|---|---|
| **MTFL** (arXiv 2410.05900) | **0.8978** | **Video Swin Transformer**, multi-timescale tubelets of 8 / 32 / 64 frames | `github.com/erktkdg/MTFL`, **no license stated** | Highest reported UCF-Crime number found, and the mechanism is orthogonal to everything we have: our heads all consume ONE timescale. It also ships its VST features (see §2). |
| **PEL4VAD** (`github.com/yujiangpu20/PEL4VAD`) | 0.8676 | I3D + text prompts | MIT | Prompt-enhanced context on **the same I3D features we already have**, so it is the cheapest new head to add — no new extraction at all. MIT license makes reuse clean. |
| **STPrompt** (arXiv 2408.05905) | 0.8808 | spatio-temporal prompt learning, no CLIP features needed | check | Beats CLIP-based methods without CLIP features; also does localization, which Spec 3 wants. |

PEL4VAD is the obvious first add: MIT, and it needs zero new features.

## 2. A second zero-GPU feature source: Video Swin (MTFL)

MTFL publishes `VST_RGB` and `VSTAug_RGB` features for UCF-Crime, XD-Violence, ShanghaiTech
and their extended VADD set. That is a **second modern backbone available without touching
the card**, alongside the LanguageBind set in `docs/FEATURE_SOURCES.md`. Video Swin is a
different family from both I3D (CNN) and VideoMAE/Cosmos (ViT), so it widens the
backbone axis rather than duplicating it.

Unknowns to resolve before fetching: exact dimension, crop protocol, snippet stride, and
the missing license. Hosted on SharePoint, not Hugging Face, so the fetcher's registry
needs a plain-URL source type — a small extension to
`scripts/fetch_pretrained_features.py`.

## 3. Prior art Spec 2 is actually competing with

Worth knowing before claiming a streaming result:

- **Real-Time WSVAD** (Karim et al., WACV 2024) — 0.8694 AUC with a **6.4 s decision
  period**, against baselines that need up to 273 s. This is the closest published
  comparison to `StreamingScorer`, and the number to beat or at least situate against.
- **Cerberus** (arXiv 2510.16290) — real-time VAD via cascaded vision-language models; a
  cheap model gates an expensive VLM. Different lever from ours (we make one model cheap
  rather than cascading two), so it is a complement in the related-work story.
- **"Anomalies are Streaming"** (continual learning for WSVAD) — streaming in the
  *training-distribution* sense, not the inference sense. Different problem; do not
  conflate it with our latency work.

Framing that follows: our contribution is not "real-time WSVAD exists" — it does — but the
**measured AUC-vs-budget curve** for a fixed, checkpoint-faithful model
(`docs/SPEC2_EFFICIENCY.md`: 0.8400 @ W=32 → 0.8697 @ full band) plus a streaming path
proven bit-identical to offline scoring. The efficiency frontier is the novel artifact.

## 4. NVIDIA / modern-stack status

- **Cosmos-Embed1** — integrated (`src/features/cosmos.py`), cost measured, leakage guard
  in place. Blocked on GPU hours, not on code.
- **NeMo** — a framework for *training* video foundation models. Wrong scale for one 8 GB
  card; revisit only if compute changes.
- **DeepStream / VSS** — deployment plumbing (RTSP ingest, multi-stream batching). Would
  matter for a demo of the streaming scorer, not for the research result.
- **TensorRT** — the one NVIDIA piece that could pay off soon: the windowed path currently
  *loses* throughput because a masked SDPA gives up the flash kernel
  (`docs/SPEC2_EFFICIENCY.md`). Either `torch.nn.attention.flex_attention` (already in
  torch 2.11, block-sparse masks, no export step) or a TRT engine could recover it.
  FlexAttention is the cheaper experiment and should be tried first.

## Suggested order (cheapest-first, GPU-aware)

1. **PEL4VAD head** on existing I3D features — new head, no new features, MIT. Pure code.
2. **LanguageBind features** (`--source languagebind`) — 26 GB download, zero GPU, then the
   forensic screen. Already scripted.
3. **FlexAttention** for the window path — recovers the throughput regression, CPU to write,
   short GPU run to verify.
4. **MTFL VST features + head** — needs the license question answered and a URL-source
   fetcher.
