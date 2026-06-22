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
    """Build (train: {normal, abnormal}, test) — local-first, HF fallback.

    ``data.source``: ``local`` forces ``~/data/wsad``; ``hub`` forces the HF cache;
    ``auto`` (default) uses local when present for the chosen ``data.backbone``,
    else HF. CLIP backbones are local-only (no HF CLIP cache yet); see
    ``docs/DATA_LOCAL.md``.
    """
    from src.data.local import build_datasets_local, has_local, has_local_i3d_zip

    source = getattr(data_cfg, "source", "hub")
    backbone = getattr(data_cfg, "backbone", "i3d")
    root = getattr(data_cfg, "root", "~/data/wsad")

    local_present = (
        backbone == "clip"
        or has_local(root, backbone, "train", data_cfg)
        or (backbone == "i3d" and has_local_i3d_zip(root, data_cfg, "train"))
    )
    use_local = source == "local" or (source == "auto" and local_present)
    if use_local:
        return build_datasets_local(data_cfg)

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
        grad_clip_norm: Optional[float] = None,  # e.g. 1.0 for BN-WVAD (official train.py:16)
        optimizer_name: str = "adam",  # "adam" | "adamw" (VadCLIP uses AdamW)
        scheduler_milestones: Optional[list] = None,  # MultiStepLR epochs, e.g. [4, 8] (VadCLIP)
        scheduler_gamma: float = 0.1,
    ):
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.model = model
        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name.lower()]
        self.optimizer = opt_cls(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # optional per-epoch LR schedule (faithful recipes: VadCLIP MultiStepLR[4,8] x0.1).
        # Stepped once per epoch in fit(); ignored by the step-based fit_steps path.
        self.scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=list(scheduler_milestones), gamma=scheduler_gamma
            )
            if scheduler_milestones
            else None
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frames_per_clip = frames_per_clip
        self.grad_clip_norm = grad_clip_norm
        sig = inspect.signature(model.forward).parameters
        self._accepts_class = "class_labels" in sig
        # heads that mask padded frames by per-video length (VadCLIP windowed-256).
        self._accepts_lengths = "lengths" in sig

    @staticmethod
    def _valid_lengths(feature: torch.Tensor) -> torch.Tensor:
        """Per-video valid (non-padded) length from a windowed feature.

        Clip windows are zero-padded to ``visual_length``; the real length is the
        count of non-zero frames (official ``process_feat`` returns this length).
        ``feature`` is ``(B, ncrops, T, D)`` or ``(B, T, D)``.
        """
        x = feature[:, 0] if feature.dim() == 4 else feature  # (B, T, D)
        return (x.abs().sum(-1) > 0).sum(-1).clamp(min=1)  # (B,)

    def fit(
        self,
        train_datasets: Dict[str, Dataset],
        test_dataset: Optional[Dataset] = None,
        epochs: int = 1,
        eval_fn=None,
        select_metric: str = "roc_auc",
        eval_every: Optional[int] = None,
    ):
        """Epoch-based training (faithful recipes that count epochs + use an LR
        schedule, e.g. VadCLIP: AdamW + MultiStepLR[4,8], 10 epochs).

        ``eval_fn(model) -> {roc_auc, pr_auc}`` (e.g. a per-head
        ``src.eval_matrix.evaluate`` closure) overrides the default I3D-shaped
        ``self.evaluate``; the BEST checkpoint by ``select_metric`` is returned.
        ``eval_every`` (optimizer steps) adds intra-epoch evals + best-checkpoint
        selection — VadCLIP's official eval at ``step % 1280 == 0`` (~12x/epoch on
        the 16100-sample 10-crop train), which catches a higher peak than 1/epoch.
        """
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

        best = {"roc_auc": 0.0, "pr_auc": 0.0, "best_epoch": -1, "last": 0.0, "_sel": -1.0}

        def _record(epoch):  # eval + keep best by select_metric
            m = eval_fn(self.accelerator.unwrap_model(model))
            best["last"] = m["roc_auc"]
            sel = m.get(select_metric, m["roc_auc"])
            if sel > best["_sel"]:
                best.update(roc_auc=m["roc_auc"], pr_auc=m["pr_auc"], best_epoch=epoch, _sel=sel)
            model.train()
            return m

        gstep = 0
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
                if self._accepts_lengths:
                    kwargs["lengths"] = self._valid_lengths(feature)
                out = model(
                    video=feature,
                    abnormal_labels=ab["anomaly"],
                    normal_labels=nb["anomaly"],
                    **kwargs,
                )
                self.accelerator.backward(out.loss)
                if self.grad_clip_norm is not None:
                    self.accelerator.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                total += out.loss.item()
                gstep += 1
                if eval_fn is not None and eval_every and gstep % eval_every == 0:
                    _record(epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            log = {"epoch": epoch, "train_loss": total / max(len(nl), 1)}
            if eval_fn is not None:
                m = _record(epoch)
                log.update(roc_auc=round(m["roc_auc"], 4), pr_auc=round(m["pr_auc"], 4),
                           best=round(best["roc_auc"], 4))
            elif test_dataset is not None:
                log.update(self.evaluate(test_dataset))
            self.accelerator.print(log)
        best.pop("_sel", None)
        return best if eval_fn is not None else log

    def fit_steps(
        self,
        train_datasets: Dict[str, Dataset],
        max_steps: int,
        eval_fn,
        eval_interval: int,
        eval_start: int = 0,
        select_metric: str = "roc_auc",
        lr_decay: Optional[str] = None,
    ) -> Dict[str, float]:
        """Iteration-based training (the WSVAD convention: RTFM/S3R/UR-DMU/... count
        gradient *steps*, not epochs). Cycles the normal/abnormal loaders, takes
        ``max_steps`` balanced mini-batches, calls ``eval_fn(model) -> {roc_auc,
        pr_auc}`` every ``eval_interval`` steps (after ``eval_start``), and keeps the
        BEST checkpoint by ``select_metric`` (AUC, or 'pr_auc' for BN-WVAD's AP).

        ``eval_fn`` is injected so the trainer stays decoupled from any specific eval
        (the matrix passes a per-head ``src.eval_matrix.evaluate`` closure).

        ``lr_decay="cosine"`` adds a per-step CosineAnnealingLR over ``max_steps`` —
        the official constant-lr RTFM recipe destabilizes (train loss climbs, AUC
        oscillates around an early lucky peak); decaying the lr lets it CONVERGE so
        the eval AUC settles high instead of bouncing. Returns ``auc_mean``/``auc_std``
        over all eval points (the honest stability metric, alongside the optimistic
        best-test-AUC the field reports).
        """
        loaders = {
            k: DataLoader(train_datasets[k], batch_size=self.batch_size, shuffle=True,
                          drop_last=True, num_workers=self.num_workers,
                          collate_fn=_collate_train)
            for k in ("normal", "abnormal")
        }
        model, optimizer, nl, al = self.accelerator.prepare(
            self.model, self.optimizer, loaders["normal"], loaders["abnormal"]
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
            if lr_decay == "cosine"
            else None
        )

        def _cycle(dl):
            while True:
                for x in dl:
                    yield x

        nit, ait = _cycle(nl), _cycle(al)
        best = {"roc_auc": 0.0, "pr_auc": 0.0, "best_step": -1, "last": 0.0, "_sel": -1.0}
        eval_aucs = []
        run_loss = 0.0
        for step in range(1, max_steps + 1):
            model.train()
            nb, ab = next(nit), next(ait)
            feature = torch.cat([nb["feature"], ab["feature"]], dim=0)
            kwargs = {}
            if self._accepts_class:
                kwargs["class_labels"] = torch.cat([nb["class_id"], ab["class_id"]], dim=0)
            if self._accepts_lengths:
                kwargs["lengths"] = self._valid_lengths(feature)
            out = model(video=feature, abnormal_labels=ab["anomaly"],
                        normal_labels=nb["anomaly"], **kwargs)
            self.accelerator.backward(out.loss)
            if self.grad_clip_norm is not None:
                self.accelerator.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            run_loss += out.loss.item()
            if step >= eval_start and (step % eval_interval == 0 or step == max_steps):
                m = eval_fn(self.accelerator.unwrap_model(model))
                best["last"] = m["roc_auc"]
                eval_aucs.append(m["roc_auc"])
                sel = m.get(select_metric, m["roc_auc"])
                if sel > best["_sel"]:
                    best.update(roc_auc=m["roc_auc"], pr_auc=m["pr_auc"], best_step=step, _sel=sel)
                self.accelerator.print(
                    {"step": step, "lr": round(optimizer.param_groups[0]["lr"], 6),
                     "train_loss": round(run_loss / eval_interval, 4),
                     "roc_auc": round(m["roc_auc"], 4), "best": round(best["roc_auc"], 4)})
                run_loss = 0.0
        best.pop("_sel", None)
        # honest stability metric: mean/std over the LATE half of eval points
        late = eval_aucs[len(eval_aucs) // 2:] or eval_aucs
        if late:
            import statistics
            best["auc_mean"] = round(statistics.mean(late), 4)
            best["auc_std"] = round(statistics.pstdev(late), 4) if len(late) > 1 else 0.0
        return best

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
