import lightning as L
from torch import nn, optim
import torch
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models import ResNet34_Weights, resnet34

class RexNet(L.LightningModule):
    def __init__(self, config, num_classes: int = 5, lora=False, rank=5):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = 'RexNet18'
        self.config = config

        weights = ResNet34_Weights.DEFAULT
        self.base_tfms = weights.transforms()
        model = resnet34(weights=weights)
        
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        self.model = model

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad = False
        
        if not lora:
            self.layers_to_finetune = [layer.strip() for layer in config.layers_to_finetune.keys()]
            for n, p in model.named_parameters():
                if n.startswith(tuple(self.layers_to_finetune)):   
                    p.requires_grad = True
        else:
            from lora_pytorch import LoRA
            model = LoRA.from_module(model, rank=rank)
                
        # Feature extractor
        self.latent_rap = nn.Sequential(*list(model.children())[:-1])
                      
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
    
    def get_latent_rapresentation_batch(self, batch, return_target = False):
        x, y = batch
        rapresentations = self.get_latent_rapresentation(x)
        
        if return_target:
            return rapresentations, y
        else:
            return rapresentations
        
    def get_latent_rapresentation(self, x):
        rapresentations = self.latent_rap(x)
        return rapresentations.squeeze(-1).squeeze(-1)
    
    def predict_from_latent(self, embeddings):
        return self.model.fc(embeddings)
    
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
        for layer, hyperparms in self.config.layers_to_finetune.items():
            named_group_params = [
                (name, p) for name, p in self.model.named_parameters()
                if name.startswith(layer) and p.requires_grad
            ]
            if not named_group_params:
                raise ValueError(f"No parameters matched for layer prefix '{layer}'")

            weights_group_params = []
            no_decay_group_params = []
            for name, p in named_group_params:
                if (
                    name.endswith('bias')
                    or 'bn' in name.lower()
                    or 'norm' in name.lower()
                ):
                    no_decay_group_params.append(p)
                else:
                    weights_group_params.append(p)
            if weights_group_params:
                param_groups.append({
                    'params': weights_group_params,
                    'lr': float(hyperparms['lr']),
                    'weight_decay': float(hyperparms['decay']),
                })
            if no_decay_group_params:
                param_groups.append({
                    'params': no_decay_group_params,
                    'lr': float(hyperparms['lr']),
                    'weight_decay': 0.0,
                })

        opt = optim.AdamW(param_groups)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.33, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
