import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

class EfficentRex(L.LightningModule):
    def __init__(self, config, num_classes: int = 5, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = 'EfficentRex'
        
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.base_tfms = weights.transforms()
        model =  efficientnet_v2_s(weights=weights)
        
        num_input_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_input_features, num_classes)
        self.model =  model
        
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        
        self.model.eval()
        self.model.classifier.train()

        self.lr = config.lr if hasattr(config, 'lr') else 1e-3
        self.best_val_loss = {'loss' : float('inf'), 
                              'epoch': 0}
        
    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        opt = optim.Adam(trainable_params, lr=self.lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.33, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
