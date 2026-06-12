"""Accelerate-based training loop for the unified WSVAD framework.

Replaces the Lightning ``runner.py`` with an explicit loop (see
``docs/TRAINING.md`` for the rationale): the normal+abnormal dual loader is just
``zip(normal, abnormal)``, AMP/device come from `accelerate`, and the loop stays
fully inspectable for the weight-equivalence work. Dataset-agnostic so it can be
driven by the HF feature cache or by synthetic data in tests.

Models that accept a ``class_labels`` kwarg (VadCLIP, GS-MoE) receive a ``(B,)``
integer per-video class id (0=Normal, 1..13 anomaly); others are unaffected.
"""

import inspect
from typing import Dict, Optional

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader, Dataset

from src.data import build_feature_dataset


def _collate_train(batch):
    return {
        "feature": torch.stack(
            [torch.as_tensor(b["feature"], dtype=torch.float32) for b in batch]
        ),
        "anomaly": torch.stack([torch.as_tensor(b["anomaly"]) for b in batch]),
        "class_id": torch.as_tensor(
            [int(b["class_id"]) for b in batch], dtype=torch.long
        ),
    }


def build_datasets(data_cfg) -> tuple:
    """Build (train: {normal, abnormal}, test) from the HF feature cache."""
    train = build_feature_dataset(
        mode="train",
        revision=getattr(data_cfg, "revision", "main"),
        cache_dir=getattr(data_cfg, "cache_dir", None),
        dynamic_load=getattr(data_cfg, "dynamic_load", True),
    )
    test = build_feature_dataset(
        mode="test",
        revision=getattr(data_cfg, "revision", "main"),
        cache_dir=getattr(data_cfg, "cache_dir", None),
        dynamic_load=getattr(data_cfg, "dynamic_load", True),
    )
    return train, test


class WSVADTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        batch_size: int = 32,
        num_workers: int = 2,
        frames_per_clip: int = 16,
        mixed_precision: str = "no",  # "no" | "fp16" | "bf16"
    ):
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frames_per_clip = frames_per_clip
        self._accepts_class = (
            "class_labels" in inspect.signature(model.forward).parameters
        )

    def fit(
        self,
        train_datasets: Dict[str, Dataset],
        test_dataset: Optional[Dataset] = None,
        epochs: int = 1,
    ):
        loaders = {
            k: DataLoader(
                train_datasets[k],
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                collate_fn=_collate_train,
            )
            for k in ("normal", "abnormal")
        }
        model, optimizer, nl, al = self.accelerator.prepare(
            self.model, self.optimizer, loaders["normal"], loaders["abnormal"]
        )

        for epoch in range(epochs):
            model.train()
            total = 0.0
            for nb, ab in zip(nl, al):
                feature = torch.cat(
                    [nb["feature"], ab["feature"]], dim=0
                )  # normal-first
                kwargs = {}
                if self._accepts_class:
                    kwargs["class_labels"] = torch.cat(
                        [nb["class_id"], ab["class_id"]], dim=0
                    )
                out = model(
                    video=feature,
                    abnormal_labels=ab["anomaly"],
                    normal_labels=nb["anomaly"],
                    **kwargs,
                )
                self.accelerator.backward(out.loss)
                optimizer.step()
                optimizer.zero_grad()
                total += out.loss.item()

            log = {"epoch": epoch, "train_loss": total / max(len(nl), 1)}
            if test_dataset is not None:
                log.update(self.evaluate(test_dataset))
            self.accelerator.print(log)
        return log

    @torch.no_grad()
    def evaluate(self, test_dataset: Dataset) -> Dict[str, float]:
        from src.inference import score_feature

        model = self.accelerator.unwrap_model(self.model).eval()
        device = str(self.accelerator.device)

        preds, labels = [], []
        for i in range(len(test_dataset)):
            s = test_dataset[i]
            p = score_feature(model, s["feature"], self.frames_per_clip, device)
            y = np.asarray(s["label"])
            n = min(len(p), len(y))
            preds.append(p[:n])
            labels.append(y[:n])
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        fpr, tpr, _ = roc_curve(labels, preds)
        precision, recall, _ = precision_recall_curve(labels, preds)
        return {
            "roc_auc": float(auc(fpr, tpr)),
            "pr_auc": float(auc(recall, precision)),
        }

    def save(self, path: str):
        self.accelerator.wait_for_everyone()
        model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(model.state_dict(), path)
