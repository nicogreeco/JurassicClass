import lightning as L
from torch import nn, optim
import torch
from torchvision.models import ResNet18_Weights, resnet18

class RexNet(L.LightningModule):
    def __init__(self, config, num_classes: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = 'RexNet18'

        weights = ResNet18_Weights.DEFAULT
        self.base_tfms = weights.transforms()
        model = resnet18(weights=weights)
        
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        self.model = model

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.fc.parameters():
            p.requires_grad = True

        self.lr = getattr(config, "lr", 1e-3)

    def on_train_epoch_start(self):
        self.model.eval()
        self.model.fc.train()   # only head trains

    def on_validation_epoch_start(self):
        self.model.eval()       # eval is fine for val; fc train() not needed

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc",  acc,  prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc",  acc,  prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        params = (p for p in self.model.parameters() if p.requires_grad)
        opt = optim.Adam(params, lr=self.lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.33, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
