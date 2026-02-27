from os import name

import lightning as L
from torch import nn, optim
import torch
import torch.nn.functional as F
from torchgen import model
from torchvision.models import vit_b_16, ViT_B_16_Weights
# from torchvision.models import vit_l_32, ViT_L_32_Weights

class ViTRex_FullFT(L.LightningModule):
    def __init__(self, config, num_classes: int = 5):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.model_name = "ViTRex_FullFT"
        self.config = config

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.base_tfms = weights.transforms()

        model = vit_b_16(weights=weights)

        # Replace head
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        self.model = model

        # Optional warmup: epoch 0 only fc trains (handled in on_train_epoch_start)
        self.warmup_fc_only = bool(getattr(config, "warmup_fc_only", True))

    def _set_trainable(self, train_fc: bool, train_backbone: bool):
        # freeze all
        for p in self.model.parameters():
            p.requires_grad = False

        # unfreeze head
        if train_fc:
            for name, p in self.model.named_parameters():
                if ".head." in name:
                    p.requires_grad = True

        # unfreeze backbone
        if train_backbone:
            for name, p in self.model.named_parameters():
                if ".head." not in name:
                    p.requires_grad = True

    def on_train_epoch_start(self):
        if self.current_epoch <= 3 and self.warmup_fc_only:
            # epoch 0: head only, backbone in eval (BN frozen)
            self._set_trainable(train_fc=True, train_backbone=False)
            self.model.eval()
            self.model.heads.train()
        else:
            # epoch 1+: full fine-tune
            self._set_trainable(train_fc=True, train_backbone=True)
            self.model.train()
        
        print(f"Epoch {self.current_epoch} - Trainable parameters:")
        for name, p in self.model.named_parameters():
                if p.requires_grad:
                    print(name)

    def forward(self, x):
        return self.model(x)

    def get_latent_rapresentation_batch(self, batch, return_target=False):
        x, y = batch
        reps = self.get_latent_rapresentation(x)
        return (reps, y) if return_target else reps

    def get_latent_rapresentation(self, x):
        reps = self.latent_rap(x)
        return reps.squeeze(-1).squeeze(-1)

    def predict_from_latent(self, embeddings):
        return self.model.heads(embeddings)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        
        if torch.cuda.is_available() and self.device.type == "cuda":
            mem_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            self.log("gpu_mem_mb", mem_mb, prog_bar=True, on_step=True, on_epoch=False)        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        """
        Expects config.layers_to_finetune with two keys:
          - fc: {lr: ..., decay: ...}
          - backbone: {lr: ..., decay: ...}
        Example:
          layers_to_finetune:
            fc: {lr: 5e-3, decay: 1e-4}
            backbone: {lr: 5e-4, decay: 1e-2}
        """
        if not hasattr(self.config, "layers_to_finetune"):
            raise ValueError("config.layers_to_finetune is required (fc + backbone groups).")

        if "fc" not in self.config.layers_to_finetune or "backbone" not in self.config.layers_to_finetune:
            raise ValueError("layers_to_finetune must contain keys: 'fc' and 'backbone'.")

        fc_hp = self.config.layers_to_finetune["fc"]
        bb_hp = self.config.layers_to_finetune["backbone"]

        def split_decay(params_with_names):
            wd_params, no_wd_params = [], []
            for name, p in params_with_names:
                if not p.requires_grad:
                    continue
                if name.endswith("bias") or "bn" in name.lower() or "norm" in name.lower():
                    no_wd_params.append(p)
                else:
                    wd_params.append(p)
            return wd_params, no_wd_params

        fc_named = [(n, p) for n, p in self.model.named_parameters() if ".head." in n]
        bb_named = [(n, p) for n, p in self.model.named_parameters() if ".head." not in n]

        fc_wd, fc_no_wd = split_decay(fc_named)
        bb_wd, bb_no_wd = split_decay(bb_named)

        param_groups = []
        if fc_wd:
            param_groups.append({"params": fc_wd, "lr": float(fc_hp["lr"]), "weight_decay": float(fc_hp["decay"])})
        if fc_no_wd:
            param_groups.append({"params": fc_no_wd, "lr": float(fc_hp["lr"]), "weight_decay": 0.0})

        if bb_wd:
            param_groups.append({"params": bb_wd, "lr": float(bb_hp["lr"]), "weight_decay": float(bb_hp["decay"])})
        if bb_no_wd:
            param_groups.append({"params": bb_no_wd, "lr": float(bb_hp["lr"]), "weight_decay": 0.0})

        opt = optim.AdamW(param_groups)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.33, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}