import os
import copy
import csv
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
from omegaconf import OmegaConf

from torchvision import datasets, transforms
from torchvision.models import EfficientNet_V2_S_Weights, ResNet34_Weights
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from torch.utils.data import DataLoader, Subset

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from utils import letterbox_to_square
from EfficentRex import EfficentRex
from RexNet import RexNet


# ===================== EVALUATION =====================

def evaluate_model(model, dataloader, idx_to_class, device="cpu"):
    model.eval()
    model.to(device)

    results = {
        "scores": [],
        "predicted_names": [],
        "real_names": [],
        "correct": [],
        "overall_accuracy": None,
    }

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x).softmax(1)
            predicted_class_id = pred.argmax(1)

            idx = torch.arange(len(predicted_class_id), device=pred.device)
            score = pred[idx, predicted_class_id]

            correct = (predicted_class_id == y).float()

            pred_cpu = predicted_class_id.cpu()
            y_cpu = y.cpu()
            score_cpu = score.cpu()
            correct_cpu = correct.cpu()

            results["scores"].extend(score_cpu.tolist())
            results["predicted_names"].extend([idx_to_class[int(c)] for c in pred_cpu])
            results["real_names"].extend([idx_to_class[int(c)] for c in y_cpu])
            results["correct"].extend(correct_cpu.tolist())

            total_correct += int(correct_cpu.sum().item())
            total_samples += len(y_cpu)

    results["overall_accuracy"] = total_correct / total_samples if total_samples > 0 else 0.0

    return results


# ===================== TRAINING =====================

def run_train(model, train_loader, val_loader, config, out_fold, in_fold):
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=8
    )

    ckpt_dir = f"./models/cross_validation/{config.experiment_name}/{config.model.name}/fold_{out_fold}.{in_fold}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    tb_logger = TensorBoardLogger(
        save_dir=f"./models/cross_validation/{config.experiment_name}/{config.model.name}",
        name="tb_logs",
        version=f"fold_{out_fold}.{in_fold}",
        default_hp_metric=False
    )

    trainer = Trainer(
        default_root_dir=f"./models/cross_validation/{config.experiment_name}/{model.model_name}/fold_{out_fold}.{in_fold}",
        logger=tb_logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=config.training.max_epochs,
        enable_checkpointing=True,
        log_every_n_steps=10
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Salva il config usato per questa run
    os.makedirs(ckpt_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(ckpt_dir, "config.yaml"))

    return checkpoint_callback.best_model_path

def run_test(model_class, ckpt_path, test_loader, idx_to_class, config, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class.load_from_checkpoint(
        ckpt_path,
        config=config,
        strict=False
    )

    results = evaluate_model(model, test_loader, idx_to_class, device=device)

    ckpt_dir = os.path.dirname(ckpt_path)
    pkl_path = os.path.join(ckpt_dir, "test_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    return results


# ===================== OUTER / INNER LOOP =====================

def run_outer_loop(train_ds, val_ds, base_cfg, network="EfficentRex", n_out=1, n_in=1):
    """
    n_out, n_in sono il massimo indice di fold:
    - n_out = 1 → outer fold 0 e 1 (2 outer fold)
    - n_in  = 1 → inner fold 0 e 1 (2 inner fold per ogni outer)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold_out = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_by_params = {}
    experiment_name = "grid_search_weight_decay"

    for out_fold, (train_idx_out, test_idx_out) in enumerate(kfold_out.split(train_ds, train_ds.targets)):
        if out_fold > n_out:
            break

        print(f"============ OUT FOLD {out_fold} ============")
        train_subset_out = Subset(train_ds, train_idx_out)
        test_subset_out = Subset(val_ds, test_idx_out)

        outer_train_labels = [train_ds.targets[i] for i in train_idx_out]
        kfold_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

        for in_fold, (train_idx_in, val_idx_in) in enumerate(kfold_in.split(train_subset_out, outer_train_labels)):
            if in_fold > n_in:
                break

            print(f"\t============ IN FOLD {in_fold} ============")

            train_indices = [train_idx_out[i] for i in train_idx_in]
            val_indices = [train_idx_out[i] for i in val_idx_in]

            train_subset_in = Subset(train_ds, train_indices)
            val_subset_in = Subset(val_ds, val_indices)

            base_cfg = OmegaConf.load(f"config/config_{network.lower()}.yaml")

            train_loader = DataLoader(
                train_subset_in,
                batch_size=base_cfg.training.batch_size,
                shuffle=True,
                num_workers=0,
            )
            val_loader = DataLoader(
                val_subset_in,
                batch_size=base_cfg.training.batch_size,
                shuffle=False,
                num_workers=0,
            )
            test_loader = DataLoader(
                test_subset_out,
                batch_size=base_cfg.training.batch_size,
                shuffle=False,
                num_workers=0,
            )

            if network.lower() == "rexnet":
                model_class = RexNet
            elif network.lower() == "efficentrex":
                model_class = EfficentRex
            else:
                raise ValueError(f"Unknown network: {network}")

            param_grid = {
                "classifier_lr": [3e-3],
                "factor": [10],
                "first_weight": [1e-4, 1e-3],           # decay for classifier
                "second_weight": [1e-4, 1e-5],    # decay for backbone layers
            }

            idx_to_class = {idx: class_ for class_, idx in train_ds.class_to_idx.items()}

            for params in ParameterGrid(param_grid):
                lr_cls = params["classifier_lr"]
                factor = params["factor"]
                first_weight = params["first_weight"]
                second_weight = params["second_weight"]

                cfg = copy.deepcopy(base_cfg)

                layers_dict = cfg["model"]["layers_to_finetune"]
                first_layer_name = list(layers_dict.keys())[0]

                # classifier hyperparams
                layers_dict[first_layer_name]["lr"] = lr_cls
                layers_dict[first_layer_name]["decay"] = first_weight

                # backbone hyperparams
                backbone_lr = lr_cls / factor
                for layer_name in list(layers_dict.keys())[1:]:
                    layers_dict[layer_name]["lr"] = backbone_lr
                    layers_dict[layer_name]["decay"] = second_weight

                cfg["experiment_name"] = experiment_name
                cfg["model"]["name"] = (
                    f"{cfg['model']['name']}_firstlr{lr_cls:g}_fac{factor:g}"
                    f"_wd1{first_weight:g}_wd2{second_weight:g}"
                )

                print(
                    f"\t\tTraining with classifier_lr={lr_cls:g}, "
                    f"factor={factor}, backbone_lr={backbone_lr:g}, "
                    f"classifier_decay={first_weight:g}, backbone_decay={second_weight:g}"
                )

                model = model_class(config=cfg.model, num_classes=cfg.num_classes)
                model.to(device)

                ckpt_path = run_train(model, train_loader, val_loader, cfg, out_fold, in_fold)

                results = run_test(
                    model_class=model_class,
                    ckpt_path=ckpt_path,
                    test_loader=test_loader,
                    idx_to_class=idx_to_class,
                    config=cfg.model,
                    device=device,
                )

                acc = results["overall_accuracy"]
                print(f"\t\tAccuracy on test set: {acc:.4f}")

                key = (lr_cls, factor)
                acc_by_params.setdefault(key, []).append(acc)

    csv_dir = f"./models/cross_validation/{experiment_name}"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "grid_search_results.csv")

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:  # "a" = append
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["classifier_lr", "factor", "mean_test_accuracy", "std_test_accuracy", "n_runs"])

        for (lr_cls, factor), acc_list in acc_by_params.items():
            mean_acc = float(np.mean(acc_list))
            std_acc = float(np.std(acc_list))
            n_runs = len(acc_list)
            writer.writerow([lr_cls, factor, mean_acc, std_acc, n_runs])

    print(f"\nSaved grid-search summary to: {csv_path}\n")


# ===================== MAIN SCRIPT =====================

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        default="EfficentRex",
        choices=["EfficentRex", "RexNet"],
        help="network to test"
    )
    parser.add_argument(
        "--n_out",
        type=int,
        default=1,
        help="max index of the outer fold (0..n_out). n_out=1 → 2 outer fold."
    )
    parser.add_argument(
        "--n_in",
        type=int,
        default=1,
        help="max index of the inner fold (0..n_in). n_in=1 → 2 inner fold per outer."
    )
    args = parser.parse_args()

    network = args.network
    config_path = f"config/config_{network.lower()}.yaml"
    config = OmegaConf.load(config_path)

    # scegli le transforms di base a seconda della rete
    if network.lower() == "efficentrex":
        weights = EfficientNet_V2_S_Weights.DEFAULT
    elif network.lower() == "rexnet":
        weights = ResNet34_Weights.DEFAULT
    else:
        raise ValueError(f"Unknown network: {network}")

    base_tfms = weights.transforms()
    del weights
    
    root = "./dataset/dataset"

    train_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Lambda(lambda im: letterbox_to_square(im, size=config.size, fill=0)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        base_tfms,  # ToTensor + Normalize
    ])

    val_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Lambda(lambda im: letterbox_to_square(im, size=config.size, fill=0)),
        transforms.CenterCrop(224),
        base_tfms,
    ])

    train_ds = datasets.ImageFolder(root=f"{root}/train", transform=train_tfms)
    val_ds = datasets.ImageFolder(root=f"{root}/train", transform=val_tfms)

    run_outer_loop(
        train_ds=train_ds,
        val_ds=val_ds,
        base_cfg=config,
        network=network,
        n_out=args.n_out,
        n_in=args.n_in,
    )

if __name__ == "__main__":
    main()
