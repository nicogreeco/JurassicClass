import numpy as np
from argparse import ArgumentParser
from omegaconf import OmegaConf

from torchvision.models import ResNet34_Weights, EfficientNet_V2_S_Weights
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Subset
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning import Trainer, Callback
from lightning.pytorch.loggers import TensorBoardLogger

from utils import visualize_image, letterbox_to_square
from EfficentRex import EfficentRex
from RexNet import RexNet

class ValidationTracker(Callback):
    def __init__(self, verbose=False):
        self.best_val_loss = float('inf')
        self.best_val_acc = float(0.0)
        self.verbose = verbose
        
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        val_acc = trainer.callback_metrics.get('val_acc')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            if self.verbose:
                print(f"New best val_loss: {self.best_val_loss:.4f}")   
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc.item()
            if self.verbose:
                print(f"New best val_acc: {self.best_val_acc:.4f}")

def run_cross_validation(model_class, base_tfms, config):
       
    train_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Lambda(lambda im: letterbox_to_square(im, size=config.size, fill=0)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),                 # (or 244 if you really want)
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        base_tfms,                                  # ToTensor + Normalize
    ])

    val_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Lambda(lambda im: letterbox_to_square(im, size=config.size, fill=0)),
        transforms.CenterCrop(224),
        base_tfms,
    ])

    root = "./dataset/dataset"

    train_ds = datasets.ImageFolder(root=f"{root}/train", transform=train_tfms)
    val_ds = datasets.ImageFolder(root=f"{root}/train", transform=val_tfms)  # Same data, different transforms

    # Initialize KFold
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = {'loss': [], 'acc': []}
    
    # Split the dataset into 5 folds
    for fold, (train_idx, val_idx) in enumerate((kfold.split(train_ds, train_ds.targets))):
        print(f"Fold {fold}:")

        model = model_class(config=config.model, num_classes=config.num_classes)
        
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx)
        
        print(f"Train subset size: {len(train_subset)}, Validation subset size: {len(val_subset)}")
        
        train_loader = DataLoader(train_subset, batch_size=config.training.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
        
        validation_tracker = ValidationTracker()
        
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", 
            mode="min",
            patience=8)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./models/cross_validation/{config.experiment_name}/{config.model.name}/fold_{fold}",
            filename="best-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",  # Include both loss and accuracy
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
        
        tb_logger = TensorBoardLogger(
            save_dir=f"./models/cross_validation/{config.experiment_name}/{config.model.name}",  # parent dir
            name=f"tb_logs",                                            # subfolder name
            version=f"fold_{fold}",                                     # unique run per fold
            default_hp_metric=False                                      # optional: disable hp metric
        )

        trainer = Trainer(
            default_root_dir=f"./models/cross_validation/{config.experiment_name}/{model.model_name}/fold_{fold}",
            logger=tb_logger,
            callbacks=[validation_tracker, early_stopping_callback, checkpoint_callback],
            max_epochs=config.training.max_epochs,
            enable_checkpointing=True,
            log_every_n_steps=10
        )
        
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        best_val_loss = validation_tracker.best_val_loss
        fold_results['loss'].append(best_val_loss)

        best_val_acc = validation_tracker.best_val_acc
        fold_results['acc'].append(best_val_acc)
        
        print(f"Fold {fold + 1} best validation loss: {best_val_loss:.4f}, best accuracy: {best_val_acc:.4f}")
        
    # Calculate statistics
    mean_val_loss = np.mean(fold_results['loss'])
    std_val_loss = np.std(fold_results['loss'])
    
    print(f"Loss:")
    for i, loss in enumerate(fold_results['loss']):
        print(f"  Fold {i + 1}: {loss:.4f}")
    print(f"  Mean ± Std: {mean_val_loss:.4f} ± {std_val_loss:.4f}")

    mean_val_acc = np.mean(fold_results['acc'])
    std_val_acc = np.std(fold_results['acc'])

    print(f"\nAccuracy:")
    for i, acc in enumerate(fold_results['acc']):
        print(f"  Fold {i + 1}: {acc:.4f}")
    print(f"  Mean ± Std: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    
    with open(f"./models/cross_validation/{config.experiment_name}/{model.model_name}/cross_validation.log", "a") as f:
        f.write("Now the file has more content!")
        f.write(f"Loss:")
        for i, loss in enumerate(fold_results['loss']):
            f.write(f"  Fold {i + 1}: {loss:.4f}")
        f.write(f"  Mean ± Std: {mean_val_loss:.4f} ± {std_val_loss:.4f}")

        mean_val_acc = np.mean(fold_results['acc'])
        std_val_acc = np.std(fold_results['acc'])

        f.write(f"\nAccuracy:")
        for i, acc in enumerate(fold_results['acc']):
            f.write(f"  Fold {i + 1}: {acc:.4f}")
        f.write(f"  Mean ± Std: {mean_val_acc:.4f} ± {std_val_acc:.4f}")

def main(config):
    match config.model.name:
        case 'RexNet':
            model_class = RexNet
            base_tfms = ResNet34_Weights.DEFAULT.transforms()
        case 'EfficentRex':
            model_class = EfficentRex
            base_tfms = EfficientNet_V2_S_Weights.DEFAULT.transforms()
    run_cross_validation(model_class, base_tfms, config)
      
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='./config.yaml',
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
