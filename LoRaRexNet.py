import lightning as L
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from lora_pytorch import LoRA

class LoRaResNet(L.LightningModule):
    def __init__(self, config, num_classes: int = 5):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.model_name = 'LoRaRexNet'

        weights = ResNet50_Weights.DEFAULT
        self.base_tfms = weights.transforms()

        backbone = resnet50(weights=weights)
        in_features = backbone.fc.in_features

        lora_wrapped = LoRA.from_module(backbone, rank=config.rank)
        lora_wrapped.module.fc = nn.Linear(in_features, num_classes)

        self.model = lora_wrapped

    def _set_trainable(self, train_fc: bool, train_lora: bool):
        # freeze everything
        for _, p in self.model.named_parameters():
            p.requires_grad = False

        # unfreeze classifier head
        if train_fc:
            for name, p in self.model.named_parameters():
                if name.startswith("module.fc."):
                    p.requires_grad = True

        # unfreeze LoRA adapters
        if train_lora:
            for name, p in self.model.named_parameters():
                if ".lora_module." in name:
                    p.requires_grad = True

    def on_fit_start(self):
        # epoch 0: only fc
        self._set_trainable(train_fc=True, train_lora=False)
        self.model.module.eval()          
        self.model.module.fc.train()

    def on_train_epoch_start(self):
        if self.current_epoch == 1:
            # epoch 1+: fc + lora
            self._set_trainable(train_fc=True, train_lora=True)
            self.model.module.train()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _eval_step(self, batch, stage: str):
        loss, acc = self._step(batch)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return {f"{stage}_loss": loss, f"{stage}_acc": acc}

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")
        
    def configure_optimizers(self):
        lora_params = [p for n, p in self.model.named_parameters() if ".lora_module." in n]
        fc_params   = [p for n, p in self.model.named_parameters() if n.startswith("module.fc.")]

        opt = optim.AdamW(
            [{"params": fc_params, "lr": 3e-3, "weight_decay": 1e-4}, 
             {"params": lora_params, "lr": 3e-4, "weight_decay": 1e-5}]
            )
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.33, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}