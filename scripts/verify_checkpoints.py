"""Verify official MGFN / VadCLIP checkpoints against this repo's ports.

- MGFN: official pkl -> convert_official_to_hf.convert -> load into MGFNForVAD.
- VadCLIP: official pth -> direct load into VadCLIPForVAD (quantify divergence).

Run: conda run -n balaenoptera python scripts/_verify_ckpts.py
"""

import torch

import src.models  # noqa: F401  (registers models)
from scripts.convert_official_to_hf import convert
from src.models.mgfn.configuration_mgfn import MGFNConfig
from src.models.mgfn.modeling_mgfn import MGFNForVideoAnomalyDetection
from src.models.vadclip.configuration_vadclip import VadCLIPConfig
from src.models.vadclip.modeling_vadclip import VadCLIPForVideoAnomalyDetection


def sep(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


# ---------------- MGFN ----------------
sep("MGFN: official pkl -> convert() -> load_state_dict")
official = torch.load("pretrained/mgfn/mgfn_ucf.pkl", map_location="cpu", weights_only=False)
print("official keys:", len(official))
converted = convert(official)
print("converted keys:", len(converted))

model = MGFNForVideoAnomalyDetection(MGFNConfig())
model_keys = set(model.state_dict().keys())
conv_keys = set(converted.keys())
print("model param keys:", len(model_keys))

missing = model_keys - conv_keys          # in model, not provided
unexpected = conv_keys - model_keys       # provided, not in model
print("MISSING (model needs, convert didn't produce):", len(missing))
for k in sorted(missing):
    print("   -", k)
print("UNEXPECTED (convert produced, model lacks):", len(unexpected))
for k in sorted(unexpected):
    print("   +", k)

# shape check on the intersection
shape_mismatch = []
msd = model.state_dict()
for k in sorted(model_keys & conv_keys):
    if tuple(msd[k].shape) != tuple(converted[k].shape):
        shape_mismatch.append((k, tuple(msd[k].shape), tuple(converted[k].shape)))
print("SHAPE MISMATCH on intersection:", len(shape_mismatch))
for k, a, b in shape_mismatch:
    print("   !", k, "model", a, "ckpt", b)

m2, u2 = model.load_state_dict(converted, strict=False)
print(f"load_state_dict(strict=False): missing={len(m2)} unexpected={len(u2)}")
mgfn_ok = (len(m2) == 0 and len(u2) == 0 and len(shape_mismatch) == 0)
print("MGFN FULLY COMPATIBLE:", mgfn_ok)

# forward smoke test (inference layout: (B, ncrops, T, 2049))
model.eval()
with torch.no_grad():
    out = model(video=torch.randn(1, 10, 32, 2049))
print("MGFN forward scores shape:", tuple(out.scores.shape))


# ---------------- VadCLIP ----------------
sep("VadCLIP: official pth -> direct load into repo port")
vc = torch.load("pretrained/vadclip/model_ucf.pth", map_location="cpu", weights_only=False)
print("official keys:", len(vc))

# build WITHOUT downloading CLIP (legacy table path) to inspect our key surface
cfg = VadCLIPConfig(use_clip_text=False)
vmodel = VadCLIPForVideoAnomalyDetection(cfg)
ours = set(vmodel.state_dict().keys())
theirs = set(vc.keys())
print("our param keys:", len(ours), "| official keys:", len(theirs))


def group(keys):
    g = {}
    for k in keys:
        top = k.split(".")[0]
        g[top] = g.get(top, 0) + 1
    return dict(sorted(g.items()))


print("\nofficial key groups:", group(theirs))
print("our key groups     :", group(ours))

common = ours & theirs
print("\nEXACT-NAME MATCH:", len(common))
# shape check on common
vsd = vmodel.state_dict()
vc_shapeok = 0
for k in common:
    if tuple(vsd[k].shape) == tuple(vc[k].shape):
        vc_shapeok += 1
print("  of which shape-compatible:", vc_shapeok)
print("only in OURS (need, official lacks):", len(ours - theirs))
print("only in OFFICIAL (unused by our port):", len(theirs - ours))
print("\nsample our-only keys:", sorted(ours - theirs)[:12])
print("sample official-only keys:", sorted(theirs - ours)[:12])
