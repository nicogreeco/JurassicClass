import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

class EfficentRex(L.LightningModule):
    def __init__(self, config, num_classes: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = 'EfficentRex'
        self.config = config
        
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.base_tfms = weights.transforms()
        model =  efficientnet_v2_s(weights=weights)
        
        num_input_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_input_features, num_classes)
        self.model =  model
        
        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.layers_to_finetune = [layer.strip() for layer in config.layers_to_finetune.keys()]
        for n, p in model.named_parameters():
            if n.startswith(tuple(self.layers_to_finetune)):   
                p.requires_grad = True
                
        # Feature extractor
        self.latent_rap = nn.Sequential(*list(model.children())[:-1])

    def on_train_epoch_start(self):
        self.model.eval()

        # warmup - first epoch only the output layer gets trained
        if self.current_epoch == 0:
            self.model.classifier.train()
        else:
            for name, module in self.model.named_modules():
                if name.startswith(tuple(self.layers_to_finetune)):
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
        return rapresentations
    
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

        opt = optim.Adam(param_groups)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.33, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
