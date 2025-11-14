import lightning as L
from torch import nn, optim
import torch
from torchvision.models import ResNet18_Weights, resnet18

class RexNet(L.LightningModule):
    def __init__(self, config, num_classes: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = 'RexNet18'
        self.config = config

        weights = ResNet18_Weights.DEFAULT
        self.base_tfms = weights.transforms()
        model = resnet18(weights=weights)
        
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        self.model = model

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.layers_to_finetune = [layer.strip() for layer in config.layers_to_finetune.keys()]
        for n, p in model.named_parameters():
            if n.startswith(tuple(self.layers_to_finetune)):   
                p.requires_grad = True
                
    def on_train_epoch_start(self):
        self.model.eval()

        # warmup - first epoch only the output layer gets trained
        if self.current_epoch == 0:
            self.model.fc.train()
        else:
            for layer in self.layers_to_finetune:
                module = getattr(self.model, layer, None)
                if module is not None:
                    module.train()

    def forward(self, x):
        logits = self.model(x)
        return logits

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
        param_groups = []
        for layer, lr in self.config.layers_to_finetune.items():
            group_params = [
                p for name, p in self.model.named_parameters()
                if name.startswith(layer) and p.requires_grad
            ]
            if not group_params:
                raise ValueError(f"No parameters matched for layer prefix '{layer}'")

            param_groups.append({'params': group_params, 'lr': float(lr)})

        opt = optim.Adam(param_groups)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.33, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
