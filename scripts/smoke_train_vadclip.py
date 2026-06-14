"""Training-loop smoke test: VadCLIP on a small subset of LOCAL clip/train.

Proves the CLIP data path + VadCLIP forward/loss/backward run end-to-end. Uses
the offline legacy text table (use_clip_text=False) so no CLIP download is needed.
"""

import glob
import os

import numpy as np

import src.models  # noqa: F401
from src.data.features import FeatureDataset
from src.data.local import _shape_clip
from src.models.vadclip.configuration_vadclip import VadCLIPConfig
from src.models.vadclip.modeling_vadclip import VadCLIPForVideoAnomalyDetection
from src.trainer import WSVADTrainer

CLIP_TRAIN = os.path.join(
    os.path.expanduser(os.environ.get("WSAD_DATA", "~/data/wsad")),
    "ucf_crime/features/clip/train",
)
N_PER = 8


def crop0(p):
    return p.endswith("__0.npy")


files = sorted(f for f in glob.glob(f"{CLIP_TRAIN}/*.npy") if crop0(f))
normal = [f for f in files if "Normal" in os.path.basename(f)][:N_PER]
abnormal = [f for f in files if "Normal" not in os.path.basename(f)][:N_PER]
print("subset normal:", len(normal), "abnormal:", len(abnormal))
print("one raw clip shape:", np.load(normal[0]).shape, np.load(normal[0]).dtype)


def build(paths):
    vals = {}
    for p in paths:
        feat = np.load(p).astype(np.float32)
        vals[os.path.basename(p)] = _shape_clip(feat, "train", 256, None)  # (1,256,512)
    return vals


nvals, avals = build(normal), build(abnormal)
print("shaped clip feature:", next(iter(nvals.values())).shape)
train = {
    "normal": FeatureDataset(list(nvals), nvals, with_magnitude=False),
    "abnormal": FeatureDataset(list(avals), avals, with_magnitude=False),
}

cfg = VadCLIPConfig(use_clip_text=False)  # offline legacy text table
model = VadCLIPForVideoAnomalyDetection(cfg)
trainer = WSVADTrainer(model, learning_rate=2e-5, batch_size=2, num_workers=0)

log = trainer.fit(train, test_dataset=None, epochs=2)
print("\nSMOKE TRAIN OK ->", log)
assert np.isfinite(log["train_loss"]), "non-finite loss"
print("VadCLIP CLIP-path forward/loss/backward executed.")
