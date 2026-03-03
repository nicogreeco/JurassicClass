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
            
        # Per-epoch: trainable AND in optimizer
        opt_ids = self._optimizer_param_ids()
        n_tensors, n_elems = self._count_trainable_in_optimizer(opt_ids)
        self.print(f"[EPOCH {self.current_epoch}] trainable_in_opt tensors={n_tensors} params={n_elems}")

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
            self.log("gpu_mem_mb", mem_mb, prog_bar=True, on_step=True, on_epoch=False)
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
        if not hasattr(self.config, "layers_to_finetune"):
            raise ValueError("config.layers_to_finetune is required.")

        def split_decay(named_params):
            wd, no_wd = [], []
            for name, p in named_params:
                n = name.lower()
                if name.endswith("bias") or "bn" in n or "norm" in n:
                    no_wd.append(p)
                else:
                    wd.append(p)
            return wd, no_wd

        param_groups = []

        for layer, hp in self.config.layers_to_finetune.items():
            layer_l = layer.lower()

            if layer_l == "fc":
                named = [(n, p) for n, p in self.model.named_parameters() if ".head." in n.lower()]
            elif layer_l == "backbone":
                named = [(n, p) for n, p in self.model.named_parameters() if ".head." not in n.lower()]
            else:
                # specific layer keys like: "encoder_layer_8", "encoder_layer_9", ...
                # torchvision ViT params include those substrings under encoder.layers.*
                key = layer_l
                named = [(n, p) for n, p in self.model.named_parameters() if key in n.lower()]

            if not named:
                raise ValueError(f"No parameters matched for layer key '{layer}'")

            wd, no_wd = split_decay(named)
            if wd:
                param_groups.append({"params": wd, "lr": float(hp["lr"]), "weight_decay": float(hp["decay"])})
            if no_wd:
                param_groups.append({"params": no_wd, "lr": float(hp["lr"]), "weight_decay": 0.0})

        opt = optim.AdamW(param_groups)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.33, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
    
    def _optimizer_param_ids(self):
        """Return a set of Python ids for all parameters contained in the first optimizer."""
        if not hasattr(self, "trainer") or self.trainer is None:
            return set()
        if not getattr(self.trainer, "optimizers", None):
            return set()

        opt = self.trainer.optimizers[0]
        ids = set()
        for group in opt.param_groups:
            for p in group["params"]:
                ids.add(id(p))
        return ids

    def _count_optimizer_params(self, ids_set):
        """Count (a) number of tensors and (b) number of elements for params in ids_set."""
        n_tensors = 0
        n_elems = 0
        for p in self.parameters():
            if id(p) in ids_set:
                n_tensors += 1
                n_elems += p.numel()
        return n_tensors, n_elems

    def _count_trainable_in_optimizer(self, ids_set):
        """Count params that are in optimizer AND require_grad=True."""
        n_tensors = 0
        n_elems = 0
        for p in self.parameters():
            if id(p) in ids_set and p.requires_grad:
                n_tensors += 1
                n_elems += p.numel()
        return n_tensors, n_elems

    def on_train_start(self):
        # One-time: optimizer membership
        opt_ids = self._optimizer_param_ids()
        n_tensors, n_elems = self._count_optimizer_params(opt_ids)
        self.print(f"[OPT] tensors={n_tensors} params={n_elems}")