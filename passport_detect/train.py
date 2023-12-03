import hydra
import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
import mlflow
import numpy
import torch
from mlflow.models import infer_signature
from model import MyModel
from omegaconf import DictConfig

from data import MyDataModule


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.data.seed)

    data_module = MyDataModule(
        csv_path=cfg.data.name,
        target=cfg.data.target,
        val_size=cfg.data.val_size,
        seed=cfg.data.seed,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    model = MyModel(cfg)

    logger = pl_loggers.MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name, tracking_uri=cfg.mlflow.uri
    )

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accum_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        log_every_n_steps=cfg.training.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data_module)

    with mlflow.start_run(run_id=logger.run_id):
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=infer_signature(
                model_input=numpy.zeros((1, cfg.model.input_dim)),
                model_output=numpy.zeros((1, cfg.model.output_dim)),
            ),
            input_example=numpy.zeros((1, cfg.model.input_dim)),
            registered_model_name=cfg.model.name,
        )

    torch.onnx.export(
        model=model, args=torch.zeros((1, cfg.model.input_dim)), f=cfg.model.path
    )


if __name__ == "__main__":
    train()
