<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-12 | Updated: 2026-06-12 -->

# mgfn

## Purpose
Full implementation of **MGFN** (Magnitude-Contrastive Glance-and-Focus Network,
AAAI'23) for weakly-supervised video anomaly detection. The reference working
model in this repo and the template all future models should follow.

## Key Files
| File | Description |
|------|-------------|
| `modeling_mgfn.py` | All modules: `MGFNFeatureAmplifier` (eq.1–2, fuses feature + magnitude channel), `GlanceBlock`/`GlanceAttention` (global clip-level transformer), `FocusBlock`/`FocusAttention` (local self-attentional conv / MHRA), `MGFNModel` (backbone), and `MGFNForVideoAnomalyDetection` (head: top-k magnitude selection + scoring + loss assembly). |
| `configuration_mgfn.py` | `MGFNConfig(PretrainedConfig)` — dims `(64,128,1024)`, depths `(3,3,2)`, block types `(gb,fb,fb)`, `k=3` top-k, `mag_ratio=0.1`, dropout/channels/heads. |
| `__init__.py` | Re-exports `modeling_mgfn` symbols. |

## For AI Agents

### Working In This Directory
- **Input contract:** `video` shaped `(batch, ncrops=10, T, 2049)`. Channels `[:2048]` = I3D features, `[2048:]` = appended magnitude; the amplifier processes them separately then fuses (`x_f + mag_ratio * x_m`).
- **Normal/abnormal split:** in training (or when `force_split=True`) the model slices the concatenated batch at `batch_size//2 * ncrops`. Crops are mean-reduced before scoring. This mirrors how `src/runner.training_step` concatenates the two dataloaders — keep them in sync.
- **Loss** (only when labels passed) = `MGFNLoss` (BCE + magnitude contrastive) + `TemporalSmoothnessLoss` + `SparsityLoss`, all from `src/loss`.
- `scripts/convert_official_to_hf.py` maps the official MGFN checkpoint keys onto this module layout — if you rename modules here, update that converter.

### Testing Requirements
- `MGFNForVideoAnomalyDetection(MGFNConfig()).forward(video=dummy_inputs)` should run and return `MGFNVideoAnomalyDetectionOutput` with `.scores` of shape `(B, T, 1)`.
- Full run `python run.py runner=mgfn` should reach ~0.86–0.87 UCF-Crime frame AUC.

### Common Patterns
- Conv1d-over-time throughout (sequence laid out as `(B*crops, C, T)`).
- Custom `MGFNLayerNorm` (channel-dim norm) vs `nn.BatchNorm1d` in Focus blocks.
- `gb` = GlanceBlock (global), `fb` = FocusBlock (local); stage list set by `mgfn_types`.

## Dependencies

### Internal
- `src/loss` — `MGFNLoss`, `SparsityLoss`, `TemporalSmoothnessLoss`.

### External
- `torch`, `einops` (`rearrange`), `transformers` (`PreTrainedModel`, `ModelOutput`).

<!-- MANUAL: -->
