"""End-to-end trainer test on synthetic features (offline, no downloads).

Proves the Accelerate loop trains + evaluates any model through the shared
contract, including the dual normal/abnormal loader and the class_labels path.
"""

import numpy as np
import pytest

import src.models  # noqa: F401
from src.data.features import FeatureDataset
from src.trainer import WSVADTrainer

NCROPS, T, DIM = 10, 32, 2048


def _train_datasets(n=4):
    def make(names):
        values = {
            nm: np.random.randn(NCROPS, T, DIM).astype(np.float32) for nm in names
        }
        return FeatureDataset(list(names), values, open_func=None, with_magnitude=True)

    normal = [f"Normal_Videos_{i:03d}_x264_i3d.npy" for i in range(n)]
    abnormal = [
        f"{c}{i:03d}_x264_i3d.npy"
        for i, c in enumerate(["Abuse", "Arson", "Fighting", "Robbery"][:n])
    ]
    return {"normal": make(normal), "abnormal": make(abnormal)}


def _test_dataset(n=3):
    names = [f"Abuse{i:03d}_x264_i3d.npy" for i in range(n)]
    # test features are full-length (nclips, ncrops, DIM)
    values = {nm: np.random.randn(20, NCROPS, DIM).astype(np.float32) for nm in names}
    labels = {
        nm: list(np.random.randint(0, 2, size=20 * 16).astype(float)) for nm in names
    }
    return FeatureDataset(
        names, values, labels=labels, open_func=None, with_magnitude=True
    )


@pytest.mark.parametrize("runner", ["rtfm", "gs_moe"])  # plain + class_labels path
def test_train_one_epoch(runner):
    if runner == "rtfm":
        from src.models.rtfm import RTFMConfig, RTFMForVideoAnomalyDetection

        model = RTFMForVideoAnomalyDetection(RTFMConfig())
    else:
        from src.models.gs_moe import GSMoEConfig, GSMoEForVideoAnomalyDetection

        model = GSMoEForVideoAnomalyDetection(GSMoEConfig(num_experts=3))

    trainer = WSVADTrainer(model, batch_size=2, num_workers=0)
    log = trainer.fit(_train_datasets(), _test_dataset(), epochs=1)

    assert np.isfinite(log["train_loss"])
    assert 0.0 <= log["roc_auc"] <= 1.0
