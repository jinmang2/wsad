"""Unified inference & evaluation for WSVAD models.

Pure-torch (no Lightning required). Builds any registered model from a Hydra
runner config, optionally loads trained weights from a Lightning checkpoint, and
either:
  - scores a single cached feature array → per-frame anomaly scores, or
  - evaluates frame-level ROC-AUC over the UCF-Crime test set.

The data-layout conventions mirror ``src.runner``:
  - train features (seg32 hub): ``(ncrops, T, D)``
  - test  features (full hub):  ``(T, ncrops, D)``  -> permuted to ``(ncrops, T, D)``
Both carry an appended magnitude channel (D = 2049) from ``FeatureDataset``.
"""

from typing import Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import _locate, instantiate
from sklearn.metrics import auc, precision_recall_curve, roc_curve

import src.models  # noqa: F401  (triggers model registration)
from src.dataset import build_feature_dataset


def build_model(
    runner: str = "mgfn",
    checkpoint: Optional[str] = None,
    config_path: str = "../configs",
    overrides: Optional[list] = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Instantiate a model from ``configs/runner/<runner>.yaml``.

    If ``checkpoint`` (a Lightning .ckpt) is given, its ``model.*`` weights are
    loaded into the bare model.
    """
    ovr = [f"runner={runner}"] + (overrides or [])
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="default", overrides=ovr)
    model_config = instantiate(cfg.runner.model_config)
    model = _locate(cfg.runner.model_class)(model_config)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        # strip the LightningModule "model." prefix
        clean = {
            k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")
        }
        missing, unexpected = model.load_state_dict(clean or state, strict=False)
        if missing or unexpected:
            print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    return model.to(device).eval()


@torch.no_grad()
def score_feature(
    model: torch.nn.Module,
    feature: np.ndarray,
    frames_per_clip: int = 16,
    device: str = "cpu",
) -> np.ndarray:
    """Per-frame anomaly scores for a single video's cached feature.

    Args:
        feature: ``(T, ncrops, D)`` array (test-hub layout, magnitude appended).
    Returns:
        1-D array of length ``T * frames_per_clip`` with scores in [0, 1].
    """
    x = torch.as_tensor(feature, dtype=torch.float32, device=device)
    # (T, ncrops, D) -> (1, ncrops, T, D)
    x = x.permute(1, 0, 2).unsqueeze(0)
    out = model(video=x)
    clip_scores = out.scores.squeeze(0).squeeze(-1).cpu().numpy()  # (T,)
    return np.repeat(clip_scores, frames_per_clip)


@torch.no_grad()
def evaluate_auc(
    model: torch.nn.Module,
    revision: str = "main",
    cache_dir: Optional[str] = None,
    frames_per_clip: int = 16,
    device: str = "cpu",
) -> Dict[str, float]:
    """Frame-level ROC-AUC / PR-AUC over the UCF-Crime test set."""
    test_dataset = build_feature_dataset(
        mode="test", revision=revision, cache_dir=cache_dir, dynamic_load=False
    )

    all_preds, all_labels = [], []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        preds = score_feature(model, sample["feature"], frames_per_clip, device)
        labels = np.asarray(sample["label"])
        # align lengths (clip features can slightly over/under-cover frames)
        n = min(len(preds), len(labels))
        all_preds.append(preds[:n])
        all_labels.append(labels[:n])

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    fpr, tpr, _ = roc_curve(labels, preds)
    rec_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    return {"roc_auc": float(rec_auc), "pr_auc": float(pr_auc)}


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="WSVAD inference / evaluation")
    p.add_argument("--runner", default="mgfn", help="configs/runner/<name>.yaml")
    p.add_argument("--checkpoint", default=None, help="Lightning .ckpt to load")
    p.add_argument("--revision", default="main", help="feature dataset revision")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--frames_per_clip", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feature", default=None, help="score a single .npy and exit")
    args = p.parse_args()

    model = build_model(args.runner, args.checkpoint, device=args.device)

    if args.feature is not None:
        scores = score_feature(
            model, np.load(args.feature), args.frames_per_clip, args.device
        )
        print(f"frames={len(scores)} mean={scores.mean():.4f} max={scores.max():.4f}")
        return

    metrics = evaluate_auc(
        model, args.revision, args.cache_dir, args.frames_per_clip, args.device
    )
    print(
        f"[{args.runner}] ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f}"
    )


if __name__ == "__main__":
    _cli()
