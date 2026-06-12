"""Accelerate training entrypoint (the recommended path; see docs/TRAINING.md).

    python train.py runner=rtfm
    python train.py runner=gs_moe trainer.cls.max_epochs=50

Builds any registered model from its runner config and trains it on the cached
feature dataset with the explicit Accelerate loop in ``src.trainer``. The legacy
Lightning path remains in ``run.py`` / ``src/runner.py``.
"""

import hydra
import omegaconf
from hydra.utils import _locate, instantiate

import src.models  # noqa: F401  (registers all models)
from src.trainer import WSVADTrainer, build_datasets


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(args: omegaconf.DictConfig) -> None:
    config = instantiate(args.runner.model_config)
    model = _locate(args.runner.model_class)(config)

    precision = "no"
    prec = str(getattr(args.trainer, "precision", "no"))
    if "16-mixed" in prec or "fp16" in prec:
        precision = "fp16"
    elif "bf16" in prec:
        precision = "bf16"

    trainer = WSVADTrainer(
        model=model,
        learning_rate=float(args.runner.optimizer.learning_rate),
        weight_decay=float(args.runner.optimizer.weight_decay),
        batch_size=int(args.data.batch_size),
        num_workers=int(args.data.num_workers),
        frames_per_clip=int(args.data.frames_per_clip),
        mixed_precision=precision,
    )

    train_datasets, test_dataset = build_datasets(args.data)
    trainer.fit(
        train_datasets,
        test_dataset,
        epochs=int(getattr(args.trainer, "max_epochs", 1)),
    )


if __name__ == "__main__":
    main()
