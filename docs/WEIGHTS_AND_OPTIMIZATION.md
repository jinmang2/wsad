# Pretrained Weights & Optimization

Two cross-cutting concerns for every model in the framework:
1. **Weight compatibility** — load an official author checkpoint into our
   re-implementation and get **numerically identical** outputs.
2. **Optimization** — FlashAttention / fused kernels / AMP for speed on the
   RTX 2060, *without* silently breaking (1).

These pull in opposite directions (fused/flash/TF32 kernels are not bit-exact),
so the rule is: **verify equivalence in an exact configuration, then opt in to
speed for training/inference.**

---

## 1. Pretrained-weight compatibility

### When it applies
Most WSVAD heads are trained *from scratch* on cached features — there is no
backbone-style pretraining to import. Weight compatibility matters when an
author releases their **final trained checkpoint** and we want to (a) reproduce
their reported AUC exactly, or (b) sanity-check our re-implementation matches
their architecture. The existing `scripts/convert_official_to_hf.py` (MGFN) is
the reference example.

### Convention: one converter per model
For a model with an official checkpoint, add `src/models/<name>/convert.py`:

```python
def convert(official_state_dict: "OrderedDict") -> "OrderedDict":
    """Remap official key names -> this repo's module names."""
    # e.g. "stages.0.1.0" -> "backbone.layers.0.3.layer_norm"
    ...
```

Then load with `strict=True` (after remap) so any missing/unexpected key is a
loud failure, never a silent zero-init.

### Numerical-equivalence protocol (the gate)
Use `src/modules/compat.py::verify_numerical_equivalence`. Requirements for a
**bit-exact** comparison:

| Setting | Value | Why |
|---------|-------|-----|
| `model.eval()` | both sides | disables dropout / BN updates |
| `attn_impl` | `"eager"` | SDPA flash/mem-efficient kernels are not bit-exact |
| dtype | `float32` | fp16/bf16 accumulate differently |
| TF32 | **off** (`torch.backends.cuda.matmul.allow_tf32 = False`) | TF32 truncates matmul mantissa |
| seed | fixed, no sampling | UR-DMU's variational latent: compare with `mu` (skip reparam) |
| device | same | CPU vs CUDA reductions differ slightly |

Pass criterion: `max|out_ours - out_official| < 1e-5` on a shared input. Anything
larger means a real architecture/key-mapping mismatch — investigate, don't widen
the tolerance.

### Gotchas baked into this repo
- Models slice the appended magnitude channel themselves (`video[..., :feature_size]`);
  the official input may be raw 2048-d. Match the input you feed both sides.
- Crop handling differs (some official repos average crops at the *score* level;
  UR-DMU here mean-pools crops up front). For exact equivalence, align this first.
- `transformers` import shims: we use `transformers.utils.ModelOutput` (stable);
  avoid the deprecated `transformers.file_utils` path in new code.

---

## 2. Optimization

### Attention: `attn_impl` flag (every attention-based model)
`src/modules/attention.py` exposes both backends; configs carry `attn_impl`:
- **`eager`** (default): explicit `softmax(QKᵀ/√d + bias)·V`. Bit-exact → use for
  the equivalence gate and for any result you will publish as "reproduced".
- **`sdpa`**: `F.scaled_dot_product_attention`, dispatching to **FlashAttention /
  memory-efficient** kernels on the GPU. Lower memory + faster for long T; the
  additive distance bias (UR-DMU) is passed as SDPA's `attn_mask`, so biased
  attention keeps the fused-kernel speedup. ~1e-3 numerical drift vs eager.

Switch per run: `python run.py runner=ur_dmu runner.model_config.attn_impl=sdpa`.

### Kernel fusion: `torch.compile`
For the light heads here the win is modest, but free:
```python
model = torch.compile(model)  # fuses pointwise ops, reduces launch overhead
```
Apply *after* loading/verifying weights (compile doesn't change numerics enough
to matter for training, but verify equivalence in eager first). Use
`mode="reduce-overhead"` for inference, default for training.

### Mixed precision (the 6 GB linchpin)
RTX 2060 has 6 GB → train in AMP:
```bash
python run.py runner=rtfm trainer.cls.precision=16-mixed
```
Features are cached, so the backbone costs ~0 VRAM at train time; AMP on the
light head keeps everything well within 6 GB at batch 32.

### TF32 (Ampere+; the 2060 is Turing so N/A locally)
On Ampere+ GPUs, `torch.set_float32_matmul_precision("high")` enables TF32 for a
free matmul speedup — but **disable it during weight verification** (truncated
mantissa breaks bit-exactness). The 2060 (Turing) has no TF32, so this only
matters when moving to newer hardware.

### Recommended profile
| Phase | attn_impl | precision | torch.compile | TF32 |
|-------|-----------|-----------|---------------|------|
| Weight-equivalence verification | eager | fp32 | off | off |
| Training (2060) | sdpa | 16-mixed | optional | n/a |
| Inference / eval | sdpa | fp16 | reduce-overhead | n/a |
