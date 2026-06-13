"""Training-loop smoke test: MGFN on a small subset of the LOCAL train.zip.

Proves the Accelerate training path runs end-to-end on real I3D features:
build datasets -> dual normal/abnormal loader -> forward -> loss -> backward.
"""

import zipfile

import numpy as np
import torch

import src.models  # noqa: F401
from src.data.features import FeatureDataset
from src.models.mgfn.configuration_mgfn import MGFNConfig
from src.models.mgfn.modeling_mgfn import MGFNForVideoAnomalyDetection
from src.trainer import WSVADTrainer

ROOT = "/home/jinmang2/data/wsad/ucf_crime"
N_PER = 8  # videos per class for the smoke subset

z = zipfile.ZipFile(f"{ROOT}/train.zip")
infos = [i for i in z.infolist() if not i.is_dir() and i.filename.endswith(".npy")]
names = [i.filename.split("/")[-1] for i in infos]
values = {n: i for n, i in zip(names, infos)}

normal = [n for n in names if "Normal" in n][:N_PER]
abnormal = [n for n in names if "Normal" not in n][:N_PER]
print("subset normal:", len(normal), "abnormal:", len(abnormal))
print("one train feature shape:", np.load(z.open(values[normal[0]])).shape)

train = {
    "normal": FeatureDataset(normal, {n: values[n] for n in normal},
                             open_func=z.open, with_magnitude=True),
    "abnormal": FeatureDataset(abnormal, {n: values[n] for n in abnormal},
                               open_func=z.open, with_magnitude=True),
}

model = MGFNForVideoAnomalyDetection(MGFNConfig())
trainer = WSVADTrainer(model, learning_rate=1e-4, batch_size=2, num_workers=0)

log = trainer.fit(train, test_dataset=None, epochs=2)
print("\nSMOKE TRAIN OK ->", log)
assert np.isfinite(log["train_loss"]), "non-finite loss"
print("loss is finite; backward/step executed.")
