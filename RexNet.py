import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from torchvision.models import ResNet18_Weights, resnet18

class RexNet(L.LightningModule):
    def __init__(self, config, num_classes: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = 'RexNet18'
        
        weights = ResNet18_Weights.DEFAULT
        self.base_tfms = weights.transforms()
        
        model = resnet18(weights=weights)
        num_input_features = model.fc.in_features
        model.fc = nn.Linear(num_input_features, num_classes)
        self.model =  model
        
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.fc.parameters():
            p.requires_grad = True
        
        self.model.eval()
        self.model.fc.train()

        self.lr = config.lr if hasattr(config, 'lr') else 1e-3
        self.validation_metrics = {'loss':[], 'acc':[]}
        self.training_metrics = {'loss':[], 'acc':[]}

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
        self.training_metrics['loss'].append(loss)
        self.training_metrics['acc'].append(acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.validation_metrics['loss'].append(loss)
        self.validation_metrics['acc'].append(acc)

    def on_validation_epoch_end(self):
        loss = sum(self.validation_metrics['loss'])/len(self.validation_metrics['loss'])
        acc = sum(self.validation_metrics['acc'])/len(self.validation_metrics['acc'])
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return super().on_on_validation_epoch_end()

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        opt = optim.Adam(trainable_params, lr=self.lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.33, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
    


