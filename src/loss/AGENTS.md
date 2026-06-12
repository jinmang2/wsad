<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-12 | Updated: 2026-06-12 -->

# loss

## Purpose
Composable loss functions for weakly-supervised anomaly detection. Models import
these and assemble the total loss inside their `forward`. Designed to be shared
across methods (the plan's slot #5).

## Key Files
| File | Description |
|------|-------------|
| `base.py` | `TemporalSmoothnessLoss` (penalizes score jumps between adjacent clips, λ=8e-4), `SparsityLoss` (encourages sparse anomaly scores, λ=8e-3), `ContrastiveLoss` (margin-based, margin=200). |
| `mgfn.py` | `MGFNLoss` — BCE classification + magnitude contrastive (separate normal/abnormal, cluster within-class) built on `ContrastiveLoss`. |
| `__init__.py` | Re-exports all four loss classes. |

## For AI Agents

### Working In This Directory
- Each loss is an `nn.Module`; smoothness/sparsity take score tensors, `MGFNLoss` takes scores + feature magnitudes + labels.
- `MGFNLoss` imports `ContrastiveLoss` from the package (`from . import ContrastiveLoss`) — preserve `__init__.py` re-exports if you add files.
- When adding methods (plan: `ranking.py`, `magnitude_contrastive.py`, `pseudo_label.py`, `self_distill.py`, `mil_align.py`), keep them as standalone `nn.Module`s with explicit λ weights so configs can compose `loss: [...]` lists.

### Testing Requirements
- Pure functions of tensors — unit-test with small random inputs and check scalar, finite, gradient-flowing outputs.

### Common Patterns
- Each loss owns its weighting constant (λ) as a constructor default, overridable per call.
- Reductions are explicit (`sum`/`mean`); be deliberate about which, it affects scale vs the λ constants.

## Dependencies

### External
- `torch`, `torch.nn`.

<!-- MANUAL: -->
