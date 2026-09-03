"""Convert facebookresearch/video-nonlocal-net Caffe2 I3D weights -> PyTorch.

Faithful port of GowthamGottimukkala/I3D_Feature_Extraction_resnet
`utils/convert_weights.py` (which itself follows Tushar-N/pytorch-resnet3d), but
targets THIS repo's ``src.i3d.I3Res50`` (architecturally identical to Gowtham's
model minus the unused ``fc``/``drop`` classifier head, so the feature path is
unchanged). The blob->param regex mapping is reproduced verbatim.

    python scripts/convert_i3d_caffe2.py \
        pretrained/i3d/i3d_baseline_32x2_IN_pretrain_400k.pkl \
        pretrained/i3d/i3d_baseline_r50_kinetics.pth          --no-nl
    python scripts/convert_i3d_caffe2.py \
        pretrained/i3d/i3d_nonlocal_32x2_IN_pretrain_400k.pkl \
        pretrained/i3d/i3d_nonlocal_r50_kinetics.pth          --nl

The .pth is a plain state_dict loadable into ``I3Res50(use_nl=...)`` with
``strict=False`` (our model has no ``fc``; every conv/bn/nl key is filled).
"""

import argparse
import os
import pickle
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.i3d import I3Res50

# ---- blob -> pytorch-param mapping (verbatim from Gowtham convert_weights.py) ----
_DS_PAT = re.compile("res(.)_(.)_branch1_.*")
_CONV_PAT = re.compile("res(.)_(.)_branch2(.)_.*")
_NL_PAT = re.compile("nonlocal_conv(.)_(.)_(.*)_.*")
_M2NUM = dict(zip("abc", [1, 2, 3]))
_SUFFIX = {"b": "bias", "w": "weight", "s": "weight", "rm": "running_mean", "riv": "running_var"}


def _key_map(blob_keys):
    km = {
        "conv1.weight": "conv1_w",
        "bn1.weight": "res_conv1_bn_s",
        "bn1.bias": "res_conv1_bn_b",
        "bn1.running_mean": "res_conv1_bn_rm",
        "bn1.running_var": "res_conv1_bn_riv",
        "fc.weight": "pred_w",
        "fc.bias": "pred_b",
    }
    for key in blob_keys:
        m = _CONV_PAT.match(key)
        if m:
            layer, block, module = int(m.group(1)), int(m.group(2)), _M2NUM[m.group(3)]
            name = "bn" if "bn_" in key else "conv"
            nk = "layer%d.%d.%s%d.%s" % (layer - 1, block, name, module, _SUFFIX[key.split("_")[-1]])
            km[nk] = key
        m = _DS_PAT.match(key)
        if m:
            layer, block = int(m.group(1)), int(m.group(2))
            module = 0 if key[-1] == "w" else 1
            nk = "layer%d.%d.downsample.%d.%s" % (layer - 1, block, module, _SUFFIX[key.split("_")[-1]])
            km[nk] = key
        m = _NL_PAT.match(key)
        if m:
            layer, block, module = int(m.group(1)), int(m.group(2)), m.group(3)
            nk = "layer%d.%d.nl.%s.%s" % (layer - 1, block, module, _SUFFIX[key.split("_")[-1]])
            km[nk] = key
    return km


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("out")
    ap.add_argument("--nl", dest="nl", action="store_true", help="non-local model")
    ap.add_argument("--no-nl", dest="nl", action="store_false")
    ap.set_defaults(nl=False)
    args = ap.parse_args()

    blobs = pickle.load(open(args.pkl, "rb"), encoding="latin")["blobs"]
    blobs = {k: v for k, v in blobs.items() if "momentum" not in k}
    km = _key_map(blobs.keys())

    model = I3Res50(use_nl=args.nl)
    sd = model.state_dict()

    new_sd, missing, mism = {}, [], []
    for key in sd:
        if key not in km:
            missing.append(key)
            continue
        c2 = torch.from_numpy(blobs[km[key]])
        if tuple(c2.shape) != tuple(sd[key].shape):
            mism.append((key, tuple(c2.shape), tuple(sd[key].shape)))
            continue
        new_sd[key] = c2

    if mism:
        for k, a, b in mism:
            print(f"  SHAPE MISMATCH {k}: caffe2 {a} != model {b}")
        raise SystemExit("aborting on shape mismatch")
    if missing:
        # legitimately absent from caffe2: classifier head (our model has none) and
        # num_batches_tracked (a non-learnable BN counter, unused in eval mode).
        bad = [m for m in missing
               if not (m.startswith(("fc.", "drop.")) or m.endswith("num_batches_tracked"))]
        if bad:
            raise SystemExit(f"unmapped model params (would init randomly!): {bad}")

    # confirm a clean strict=False load (no leftover-random conv/bn/nl weights/buffers)
    miss, unexp = model.load_state_dict(new_sd, strict=False)
    leftover = [k for k in miss
                if not (k.startswith(("fc.", "drop.")) or k.endswith("num_batches_tracked"))]
    assert not leftover, f"params not loaded: {leftover}"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(new_sd, args.out)
    print(f"use_nl={args.nl}: mapped {len(new_sd)}/{len(sd)} params "
          f"(skipped fc/drop), {len(blobs)} blobs -> {args.out}")


if __name__ == "__main__":
    main()
