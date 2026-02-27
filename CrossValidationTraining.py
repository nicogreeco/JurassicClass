import csv
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import torch
from torchvision.models import ResNet34_Weights, EfficientNet_V2_S_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from lightning import Trainer, Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from scripts.utils import letterbox_to_square


WARMUP_EPOCHS = 4  # exclude epochs 0..3 from avg epoch time


class ValidationTracker(Callback):
    def __init__(self, verbose=False):
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.verbose = verbose

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return

        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")

        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            if self.verbose:
                print(f"New best val_loss: {self.best_val_loss:.4f}")

        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc.item()
            if self.verbose:
                print(f"New best val_acc: {self.best_val_acc:.4f}")


class GpuMemTracker(Callback):
    """Tracks a single value per fit: peak GPU memory allocated (MB)."""
    def __init__(self, verbose=False):
        self.peak_mb = float("nan")
        self.verbose = verbose

    def _get_device(self, trainer, pl_module):
        dev = getattr(getattr(trainer, "strategy", None), "root_device", None)
        if dev is None:
            dev = getattr(pl_module, "device", None)
        return dev

    def on_fit_start(self, trainer, pl_module):
        dev = self._get_device(trainer, pl_module)
        if torch.cuda.is_available() and dev is not None and dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

    def on_fit_end(self, trainer, pl_module):
        dev = self._get_device(trainer, pl_module)
        if torch.cuda.is_available() and dev is not None and dev.type == "cuda":
            self.peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
            if self.verbose:
                print(f"Peak GPU Memory Usage: {self.peak_mb:.2f} MB")


class EpochTimeTracker(Callback):
    """
    Measures per-epoch wall time (train epoch + validation) and computes
    avg epoch time excluding warmup epochs (0..warmup_epochs-1).
    """
    def __init__(self, warmup_epochs=4, verbose=False):
        self.warmup_epochs = int(warmup_epochs)
        self.verbose = verbose
        self._t0 = None

        self.epoch_times_sec_all = []      # all epochs (incl warmup)
        self.epoch_times_sec_timed = []    # epochs >= warmup_epochs only

    def on_train_epoch_start(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        self._t0 = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        # end-of-epoch timing after validation completes
        if getattr(trainer, "sanity_checking", False):
            return
        if self._t0 is None:
            return

        dt = time.perf_counter() - self._t0
        self._t0 = None

        epoch_idx = int(trainer.current_epoch)
        self.epoch_times_sec_all.append(dt)

        if epoch_idx >= self.warmup_epochs:
            self.epoch_times_sec_timed.append(dt)

        if self.verbose:
            tag = "TIMED" if epoch_idx >= self.warmup_epochs else "warmup"
            print(f"Epoch {epoch_idx} time ({tag}): {dt:.2f}s")

    @property
    def avg_epoch_time_sec_excl_warmup(self) -> float:
        if not self.epoch_times_sec_timed:
            return float("nan")
        return float(np.mean(self.epoch_times_sec_timed))

    @property
    def epochs_trained_total(self) -> int:
        return len(self.epoch_times_sec_all)

    @property
    def epochs_timed(self) -> int:
        return len(self.epoch_times_sec_timed)


def mean_std(values):
    arr = np.array(values, dtype=float)
    return np.nanmean(arr), np.nanstd(arr)


def fmt_seconds(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_cross_validation(model_class, base_tfms, config):
    train_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Lambda(lambda im: letterbox_to_square(im, size=config.size, fill=0)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        base_tfms,
    ])

    eval_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Lambda(lambda im: letterbox_to_square(im, size=config.size, fill=0)),
        transforms.CenterCrop(224),
        base_tfms,
    ])

    root = "./dataset/dataset"

    train_ds = datasets.ImageFolder(root=f"{root}/train", transform=train_tfms)
    val_ds = datasets.ImageFolder(root=f"{root}/train", transform=eval_tfms)
    test_ds = datasets.ImageFolder(root=f"{root}/test", transform=eval_tfms)

    # Ensure train/test class order matches
    if train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError(
            f"Class mapping mismatch between train and test.\n"
            f"train: {train_ds.class_to_idx}\n"
            f"test:  {test_ds.class_to_idx}"
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = {
        "val_loss": [],
        "val_acc": [],
        "test_acc": [],
        "avg_epoch_time_sec": [],     # avg epoch time excluding warmup
        "epochs_trained": [],         # total epochs actually run (incl warmup)
        "epochs_timed": [],           # epochs counted in avg (excl warmup)
        "gpu_mem_mb": [],
    }
    rows = []

    model_name = config.model.name
    base_out_dir = Path(f"./models/cross_validation/{config.experiment_name}/{model_name}")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_ds, train_ds.targets)):
        print("\n\n===========================================================")
        print(f"Fold {fold}")
        print("===========================================================")

        model = model_class(config=config.model, num_classes=config.num_classes)

        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx)

        print(f"Train subset size: {len(train_subset)}")
        print(f"Validation subset size: {len(val_subset)}")
        print(f"Test set size: {len(test_ds)}")

        train_loader = DataLoader(
            train_subset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=0,
        )

        validation_tracker = ValidationTracker()
        gpu_tracker = GpuMemTracker()
        epoch_time_tracker = EpochTimeTracker(warmup_epochs=WARMUP_EPOCHS)

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=12,
        )

        checkpoint_dir = base_out_dir / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=False,
        )

        tb_logger = TensorBoardLogger(
            save_dir=str(base_out_dir),
            name="tb_logs",
            version=f"fold_{fold}",
            default_hp_metric=False,
        )

        trainer = Trainer(
            default_root_dir=str(checkpoint_dir),
            logger=tb_logger,
            callbacks=[validation_tracker, gpu_tracker, epoch_time_tracker, early_stopping_callback, checkpoint_callback],
            max_epochs=config.training.max_epochs,
            enable_checkpointing=True,
            log_every_n_steps=10,
        )

        # -------- Train --------
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_val_loss = float(validation_tracker.best_val_loss)
        best_val_acc = float(validation_tracker.best_val_acc)
        gpu_mem_usage = float(gpu_tracker.peak_mb)

        avg_epoch_time_sec = float(epoch_time_tracker.avg_epoch_time_sec_excl_warmup)
        epochs_trained = int(epoch_time_tracker.epochs_trained_total)
        epochs_timed = int(epoch_time_tracker.epochs_timed)

        fold_results["val_loss"].append(best_val_loss)
        fold_results["val_acc"].append(best_val_acc)
        fold_results["avg_epoch_time_sec"].append(avg_epoch_time_sec)
        fold_results["epochs_trained"].append(epochs_trained)
        fold_results["epochs_timed"].append(epochs_timed)
        fold_results["gpu_mem_mb"].append(gpu_mem_usage)

        # -------- Test best checkpoint --------
        test_acc = float("nan")

        try:
            test_output = trainer.test(
                model=model,
                dataloaders=test_loader,
                ckpt_path="best",
                verbose=False,
            )

            if test_output and len(test_output) > 0:
                metrics = test_output[0]
                for acc_key in ["test_acc", "test_accuracy", "test/top1_acc", "test_top1_acc"]:
                    if acc_key in metrics:
                        test_acc = float(metrics[acc_key])
                        break

        except Exception as e:
            print(f"[WARNING] Test step failed on fold {fold}: {e}")
            print("Make sure your LightningModule defines test_step and logs 'test_acc'.")

        fold_results["test_acc"].append(test_acc)

        best_ckpt_path = checkpoint_callback.best_model_path

        # delete best checkpoint after test
        if trainer.is_global_zero and best_ckpt_path:
            p = Path(best_ckpt_path)
            if p.exists():
                p.unlink()

        rows.append({
            "fold": fold,
            "best_ckpt": best_ckpt_path,
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
            "test_acc": test_acc,
            "avg_epoch_time_sec": avg_epoch_time_sec,
            "epochs_trained": epochs_trained,
            "epochs_timed": epochs_timed,
            "gpu_mem_mb": gpu_mem_usage,
        })

        print("-----------------------------------------------------------")
        print(f"Best val loss     : {best_val_loss:.4f}")
        print(f"Best val acc      : {best_val_acc:.4f}")
        print(f"Test acc          : {test_acc:.4f}" if not np.isnan(test_acc) else "Test acc          : NaN")
        print(f"Epochs trained    : {epochs_trained} (warmup excluded from timing: {WARMUP_EPOCHS})")
        print(f"Timed epochs      : {epochs_timed}")

        if not np.isnan(avg_epoch_time_sec):
            print(f"Avg epoch time    : {fmt_seconds(avg_epoch_time_sec)} ({avg_epoch_time_sec:.1f}s)  [excl warmup]")
        else:
            print("Avg epoch time    : NaN  [excl warmup]")

        print(f"Peak GPU mem      : {gpu_mem_usage:.2f} MB" if not np.isnan(gpu_mem_usage) else "Peak GPU mem      : NaN")
        print("===========================================================")

    # -------- Summary --------
    val_loss_mean, val_loss_std = mean_std(fold_results["val_loss"])
    val_acc_mean, val_acc_std = mean_std(fold_results["val_acc"])
    test_acc_mean, test_acc_std = mean_std(fold_results["test_acc"])
    avg_epoch_time_mean, avg_epoch_time_std = mean_std(fold_results["avg_epoch_time_sec"])
    gpu_mem_mean, gpu_mem_std = mean_std(fold_results["gpu_mem_mb"])

    epochs_trained_mean, epochs_trained_std = mean_std(fold_results["epochs_trained"])
    epochs_timed_mean, epochs_timed_std = mean_std(fold_results["epochs_timed"])

    print("\n===========================================================")
    print("Cross-validation summary")
    print("===========================================================")

    print("Validation Loss:")
    for i, v in enumerate(fold_results["val_loss"]):
        print(f"  Fold {i}: {v:.4f}")
    print(f"  Mean ± Std: {val_loss_mean:.4f} ± {val_loss_std:.4f}")

    print("\nValidation Accuracy:")
    for i, v in enumerate(fold_results["val_acc"]):
        print(f"  Fold {i}: {v:.4f}")
    print(f"  Mean ± Std: {val_acc_mean:.4f} ± {val_acc_std:.4f}")

    print("\nTest Accuracy:")
    for i, v in enumerate(fold_results["test_acc"]):
        print(f"  Fold {i}: {v:.4f}" if not np.isnan(v) else f"  Fold {i}: NaN")
    print(f"  Mean ± Std: {test_acc_mean:.4f} ± {test_acc_std:.4f}")

    print(f"\nEpochs trained (total, incl warmup): mean ± std = {epochs_trained_mean:.2f} ± {epochs_trained_std:.2f}")
    print(f"Timed epochs (excl first {WARMUP_EPOCHS}): mean ± std = {epochs_timed_mean:.2f} ± {epochs_timed_std:.2f}")

    print(f"\nAvg Epoch Time (seconds) [excl first {WARMUP_EPOCHS} epochs]:")
    for i, v in enumerate(fold_results["avg_epoch_time_sec"]):
        print(f"  Fold {i}: {v:.1f}s ({fmt_seconds(v)})" if not np.isnan(v) else f"  Fold {i}: NaN")
    print(f"  Mean ± Std: {avg_epoch_time_mean:.1f}s ± {avg_epoch_time_std:.1f}s")

    print("\nPeak GPU Memory (MB):")
    for i, v in enumerate(fold_results["gpu_mem_mb"]):
        print(f"  Fold {i}: {v:.2f} MB" if not np.isnan(v) else f"  Fold {i}: NaN")
    print(f"  Mean ± Std: {gpu_mem_mean:.2f} ± {gpu_mem_std:.2f} MB")

    print("===========================================================")

    # -------- Save CSV table --------
    csv_path = base_out_dir / "cross_validation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "best_ckpt",
                "val_loss",
                "val_acc",
                "test_acc",
                "avg_epoch_time_sec",
                "epochs_trained",
                "epochs_timed",
                "gpu_mem_mb",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

        writer.writerow({
            "fold": "mean",
            "best_ckpt": "",
            "val_loss": val_loss_mean,
            "val_acc": val_acc_mean,
            "test_acc": test_acc_mean,
            "avg_epoch_time_sec": avg_epoch_time_mean,
            "epochs_trained": epochs_trained_mean,
            "epochs_timed": epochs_timed_mean,
            "gpu_mem_mb": gpu_mem_mean,
        })
        writer.writerow({
            "fold": "std",
            "best_ckpt": "",
            "val_loss": val_loss_std,
            "val_acc": val_acc_std,
            "test_acc": test_acc_std,
            "avg_epoch_time_sec": avg_epoch_time_std,
            "epochs_trained": epochs_trained_std,
            "epochs_timed": epochs_timed_std,
            "gpu_mem_mb": gpu_mem_std,
        })

    # -------- Save readable log --------
    log_path = base_out_dir / "cross_validation.log"
    with open(log_path, "a") as f:
        f.write("\n===========================================================\n")
        f.write("Cross-validation summary\n")
        f.write("===========================================================\n")

        f.write("Validation Loss:\n")
        for i, v in enumerate(fold_results["val_loss"]):
            f.write(f"  Fold {i}: {v:.4f}\n")
        f.write(f"  Mean ± Std: {val_loss_mean:.4f} ± {val_loss_std:.4f}\n")

        f.write("\nValidation Accuracy:\n")
        for i, v in enumerate(fold_results["val_acc"]):
            f.write(f"  Fold {i}: {v:.4f}\n")
        f.write(f"  Mean ± Std: {val_acc_mean:.4f} ± {val_acc_std:.4f}\n")

        f.write("\nTest Accuracy:\n")
        for i, v in enumerate(fold_results["test_acc"]):
            f.write(f"  Fold {i}: {v:.4f}\n" if not np.isnan(v) else f"  Fold {i}: NaN\n")
        f.write(f"  Mean ± Std: {test_acc_mean:.4f} ± {test_acc_std:.4f}\n")

        f.write(f"\nEpochs trained (total, incl warmup): mean ± std = {epochs_trained_mean:.2f} ± {epochs_trained_std:.2f}\n")
        f.write(f"Timed epochs (excl first {WARMUP_EPOCHS}): mean ± std = {epochs_timed_mean:.2f} ± {epochs_timed_std:.2f}\n")

        f.write(f"\nAvg Epoch Time (seconds) [excl first {WARMUP_EPOCHS} epochs]:\n")
        for i, v in enumerate(fold_results["avg_epoch_time_sec"]):
            f.write(f"  Fold {i}: {v:.1f}s ({fmt_seconds(v)})\n" if not np.isnan(v) else f"  Fold {i}: NaN\n")
        f.write(f"  Mean ± Std: {avg_epoch_time_mean:.1f}s ± {avg_epoch_time_std:.1f}s\n")

        f.write("\nPeak GPU Memory (MB):\n")
        for i, v in enumerate(fold_results["gpu_mem_mb"]):
            f.write(f"  Fold {i}: {v:.2f} MB\n" if not np.isnan(v) else f"  Fold {i}: NaN\n")
        f.write(f"  Mean ± Std: {gpu_mem_mean:.2f} ± {gpu_mem_std:.2f} MB\n")

        f.write("===========================================================\n")

    print(f"\nSaved results table to: {csv_path}")
    print(f"Saved summary log to:   {log_path}")


def main(config):
    match config.model.name:
        case "RexNet":
            from scripts.RexNet import RexNet
            model_class = RexNet
            base_tfms = ResNet34_Weights.DEFAULT.transforms()

        case "EfficentRex":
            from scripts.EfficentRex import EfficentRex
            model_class = EfficentRex
            base_tfms = EfficientNet_V2_S_Weights.DEFAULT.transforms()

        case "LoRaRexNet":
            from scripts.LoRaRexNet import LoRaResNet
            model_class = LoRaResNet
            base_tfms = ResNet34_Weights.DEFAULT.transforms()

        case "RexNet_FullFT":
            from scripts.RexNet_FullFT import RexNet_FullFT
            model_class = RexNet_FullFT
            base_tfms = ResNet34_Weights.DEFAULT.transforms()

        case "LoRaViTRex":
            from scripts.LoRaViTRex import LoRaViTRex
            model_class = LoRaViTRex
            base_tfms = ResNet34_Weights.DEFAULT.transforms()

        case "ViTRex_FullFT":
            from scripts.ViTRex import ViTRex_FullFT
            model_class = ViTRex_FullFT
            base_tfms = ResNet34_Weights.DEFAULT.transforms()

        case _:
            raise ValueError(f"Unknown model name: {config.model.name}")

    run_cross_validation(model_class, base_tfms, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)