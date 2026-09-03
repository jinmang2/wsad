# Unbounded-gradient hazards in the ported heads (audit 2026-09-03)

PEL4VAD diverged in training because `sqrt` has an **unbounded derivative at zero**
(`0.5/sqrt(x)`), and a single spike of 6.6e18 poisoned Adam's second moment so the model
never recovered (see `docs/RESULTS_TABLE.md` and the fix in `modeling_pel4vad.py`). That is
not a PEL4VAD-specific mistake — it is a property of every `sqrt`, `norm` or division that
can reach zero, and the official implementations these heads are ported from are full of
them.

This audit swept `src/models`, `src/modules` and `src/loss` for the pattern and checked
**reachability** by actually constructing the degenerate input, rather than reasoning about
whether it could happen.

## Findings

| site | expression | degenerate input | gradient |
|---|---|---|---|
| `mgfn/modeling_mgfn.py:38` `MGFNLayerNorm` | `var(x, dim=1).sqrt()`, then `/(std + eps)` | any slice constant across channels → `var == 0` | **NaN** |
| ↳ same, near-degenerate | | `var ≈ 1e-15` | 6.8e2 |
| `bn_wvad/modeling_bn_wvad.py:129` `get_mahalanobis_distance` | `sqrt(sum((f-a)²/var))`, comment says *"official, no eps"* | `feats == anchor` exactly | **NaN** |
| `bn_wvad/modeling_bn_wvad.py:224` | `sqrt(sum(...) + 1e-12)` | — | finite (already guarded) |
| `pel4vad/modeling_pel4vad.py:128` | `sign(x)·sqrt(|x| + 1e-6)` | — | finite (fixed) |

The `eps` in `MGFNLayerNorm` guards the **division**, not the `sqrt` — a distinction that is
easy to miss when reading the code, and the reason the hazard survives there. BN-WVAD is
more striking still: the *same file* already guards the sibling function at line 224 with
`+ 1e-12`, so line 129 is inconsistent with its own neighbour.

Reachability is not hypothetical for MGFN: features are ReLU outputs of I3D, so an all-zero
snippet (a padded tail, or a genuinely empty channel block) makes `var` exactly zero.

## Deliberately not fixed

MGFN reproduces 0.8332 and BN-WVAD 0.8233 today, and neither has been observed to diverge.
A guard changes the forward pass in the near-degenerate regime — clamping `var` to 1e-12
shifts `MGFNLayerNorm`'s denominator by ~6% when `var ≈ 1e-13` — so applying it now would
alter numbers that were verified against official checkpoints, in exchange for a risk that
has not materialised. PEL4VAD was fixed because its divergence was **measured**, not feared.

The rule this audit is meant to leave behind: **if MGFN or BN-WVAD ever diverges with a
climbing loss and a collapse to constant output, look here first.** The fix is one line in
each case — `var.clamp_min(1e-12)` and `+ 1e-12` inside the `sqrt` respectively — and
`tests/test_gradient_hazards.py` pins the current behaviour so a change is deliberate.
