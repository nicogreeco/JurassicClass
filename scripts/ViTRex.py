import lightning as L
from torch import nn, optim
import torch
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTRex_FullFT(L.LightningModule):
    def __init__(self, config, num_classes: int = 5):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.model_name = "ViTRex_FullFT"
        self.config = config

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.base_tfms = weights.transforms()

        model = vit_b_16(weights=weights)

        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        self.model = model

        self.warmup_fc_only = bool(getattr(config, "warmup_fc_only", True))

        # Make optimizer creation consistent
        self._set_trainable(train_fc=True, train_backbone=False if self.warmup_fc_only else True)

    def _get_backbone_selection_keys(self):
        keys = list(getattr(self.config, "layers_to_finetune", {}).keys())
        keys = [k for k in keys if k != "fc"]
        return keys

    def _set_trainable(self, train_fc: bool, train_backbone: bool):
        # freeze all
        for p in self.model.parameters():
            p.requires_grad = False

        # unfreeze head
        if train_fc:
            for name, p in self.model.named_parameters():
                if ".head." in name.lower():
                    p.requires_grad = True

        # unfreeze backbone
        if train_backbone:
            keys = self._get_backbone_selection_keys()
            keys_lower = [k.lower() for k in keys]

            if "backbone" in keys_lower:
                for name, p in self.model.named_parameters():
                    if ".head." not in name.lower():
                        p.requires_grad = True
            else:
                for name, p in self.model.named_parameters():
                    n = name.lower()
                    if any(k in n for k in keys_lower):
                        p.requires_grad = True

    def on_fit_start(self):
        if self.warmup_fc_only:
            self._set_trainable(train_fc=True, train_backbone=False)
            self.model.eval()
            self.model.heads.train()
        else:
            self._set_trainable(train_fc=True, train_backbone=True)
            self.model.train()

    def on_train_epoch_start(self):
        if self.current_epoch <= 3 and self.warmup_fc_only:
            self._set_trainable(train_fc=True, train_backbone=False)
            self.model.eval()
            self.model.heads.train()
        else:
            self._set_trainable(train_fc=True, train_backbone=True)
            self.model.train()

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

        if torch.cuda.is_available() and self.device.type == "cuda":
            mem_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
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
        if not hasattr(self.config, "layers_to_finetune"):
            raise ValueError("config.layers_to_finetune is required (fc + backbone groups).")

        if "fc" not in self.config.layers_to_finetune or "backbone" not in self.config.layers_to_finetune:
            raise ValueError("layers_to_finetune must contain keys: 'fc' and 'backbone'.")

        fc_hp = self.config.layers_to_finetune["fc"]
        bb_hp = self.config.layers_to_finetune["backbone"]

        def split_decay(params_with_names):
            wd_params, no_wd_params = [], []
            for name, p in params_with_names:
                n = name.lower()
                if name.endswith("bias") or "bn" in n or "norm" in n:
                    no_wd_params.append(p)
                else:
                    wd_params.append(p)
            return wd_params, no_wd_params

        fc_named = [(n, p) for n, p in self.model.named_parameters() if ".head." in n.lower()]
        bb_named = [(n, p) for n, p in self.model.named_parameters() if ".head." not in n.lower()]

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