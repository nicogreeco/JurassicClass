import lightning as L
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.models import ResNet101_Weights, resnet101
from lora_pytorch import LoRA


class LoRaResNet(L.LightningModule):
    def __init__(self, config, num_classes: int = 5):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.model_name = "LoRaRexNet"

        weights = ResNet101_Weights.DEFAULT
        self.base_tfms = weights.transforms()

        backbone = resnet101(weights=weights)
        in_features = backbone.fc.in_features

        lora_wrapped = LoRA.from_module(backbone, rank=config.rank)
        lora_wrapped.module.fc = nn.Linear(in_features, num_classes)
        self.model = lora_wrapped

        # Optional warmup: first epochs only head trains
        self.warmup_fc_only = bool(getattr(config, "warmup_fc_only", True))
        self._set_trainable(train_fc=True, train_lora=False)

    def _get_lora_selection_keys(self):
        keys = list(getattr(self.config, "layers_to_finetune", {}).keys())
        keys = [k for k in keys if k != "fc"]
        return keys

    def _set_trainable(self, train_fc: bool, train_lora: bool):
        # Freeze everything
        for _, p in self.model.named_parameters():
            p.requires_grad = False

        # Unfreeze head (ResNet classifier)
        if train_fc:
            for name, p in self.model.named_parameters():
                if name.startswith("module.fc."):
                    p.requires_grad = True

        # Unfreeze LoRA adapters
        if train_lora:
            keys = self._get_lora_selection_keys()
            keys_lower = [k.lower() for k in keys]

            if "backbone" in keys_lower:
                for name, p in self.model.named_parameters():
                    n = name.lower()
                    if ".lora_module." in n and ".fc." not in n:
                        p.requires_grad = True
            else:
                for name, p in self.model.named_parameters():
                    n = name.lower()
                    if ".lora_module." in n and any(k in n for k in keys_lower):
                        p.requires_grad = True

    def on_fit_start(self):
        if self.warmup_fc_only:
            self._set_trainable(train_fc=True, train_lora=False)
            self.model.module.eval()
            self.model.module.fc.train()
        else:
            self._set_trainable(train_fc=True, train_lora=True)
            self.model.module.train()

    def on_train_epoch_start(self):
        if self.current_epoch <= 1 and self.warmup_fc_only:
            self._set_trainable(train_fc=True, train_lora=False)
            self.model.module.eval()
            self.model.module.fc.train()
        else:
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
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_acc", acc, prog_bar=False, on_step=True, on_epoch=True)

        if torch.cuda.is_available() and self.device.type == "cuda":
            mem_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
            self.log("gpu_mem_mb", mem_mb, prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        def split_decay(params_with_names):
            wd_params, no_wd_params = [], []
            for name, p in params_with_names:
                if not p.requires_grad:
                    continue
                n = name.lower()
                if name.endswith("bias") or "bn" in n or "norm" in n:
                    no_wd_params.append(p)
                else:
                    wd_params.append(p)
            return wd_params, no_wd_params

        if not hasattr(self.config, "layers_to_finetune"):
            raise ValueError("config.layers_to_finetune is required (fc + backbone groups).")

        param_groups = []

        for layer, hyperparms in self.config.layers_to_finetune.items():
            layer_l = layer.lower()

            if layer_l == "fc":
                fc_named = [(n, p) for n, p in self.model.named_parameters() if ".fc." in n.lower()]
                fc_wd, fc_no_wd = split_decay(fc_named)
                if fc_wd:
                    param_groups.append(
                        {"params": fc_wd, "lr": float(hyperparms["lr"]), "weight_decay": float(hyperparms["decay"])}
                    )
                if fc_no_wd:
                    param_groups.append({"params": fc_no_wd, "lr": float(hyperparms["lr"]), "weight_decay": 0.0})

            elif layer_l == "backbone":
                bb_named = [
                    (n, p)
                    for n, p in self.model.named_parameters()
                    if ".lora_module." in n.lower() and ".fc." not in n.lower()
                ]
                bb_wd, bb_no_wd = split_decay(bb_named)
                if bb_wd:
                    param_groups.append(
                        {"params": bb_wd, "lr": float(hyperparms["lr"]), "weight_decay": float(hyperparms["decay"])}
                    )
                if bb_no_wd:
                    param_groups.append({"params": bb_no_wd, "lr": float(hyperparms["lr"]), "weight_decay": 0.0})

            else:
                key = layer_l
                named_group_params = [
                    (n, p)
                    for n, p in self.model.named_parameters()
                    if ".lora_module." in n.lower() and key in n.lower()
                ]
                if not named_group_params:
                    raise ValueError(f"No parameters matched for layer key '{layer}'")

                weights_group_params, no_decay_group_params = split_decay(named_group_params)
                if weights_group_params:
                    param_groups.append(
                        {
                            "params": weights_group_params,
                            "lr": float(hyperparms["lr"]),
                            "weight_decay": float(hyperparms["decay"]),
                        }
                    )
                if no_decay_group_params:
                    param_groups.append(
                        {"params": no_decay_group_params, "lr": float(hyperparms["lr"]), "weight_decay": 0.0}
                    )

        opt = optim.AdamW(param_groups)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.33, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}