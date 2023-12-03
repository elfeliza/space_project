from typing import Any

import lightning.pytorch as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch import softmax
from torch.nn import BatchNorm1d, CrossEntropyLoss, Linear, ReLU
from torch.optim import Adam


class MyModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.linear_1 = Linear(cfg.model.input_dim, cfg.model.hidden_dim1)
        self.linear_2 = Linear(cfg.model.hidden_dim1, cfg.model.output_dim)
        self.act_1 = ReLU()
        self.batchnorm_1 = BatchNorm1d(cfg.model.input_dim)
        self.loss_fn = CrossEntropyLoss()
        self.f1_fn = torchmetrics.classification.F1Score(
            task=cfg.model.f1_task, num_classes=cfg.model.output_dim
        )

    def forward(self, x):
        x = self.batchnorm_1(x)
        x = self.linear_1(x)
        x = self.act_1(x)
        x = self.linear_2(x)
        return softmax(x, dim=1)

    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=self.cfg.model.lr)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        inputs, labels = batch
        outputs: torch.Tensor = self(inputs)
        loss = self.loss_fn(outputs, labels)
        predicted = torch.argmax(outputs, dim=1)
        val_acc = torch.sum(labels == predicted).item() / (len(predicted) * 1.0)
        f1 = self.f1_fn(predicted, labels).item()
        self.log_dict(
            {"val_loss": loss, "val_acc": val_acc, "val_f1": f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
