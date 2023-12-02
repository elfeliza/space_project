import hydra
import mlflow
import pandas as pd
import torch
from dvc.fs import DVCFileSystem
from omegaconf import DictConfig
from torchmetrics import F1Score


@hydra.main(version_base=None, config_path="../configs", config_name="test")
def infer(cfg: DictConfig):
    fs = DVCFileSystem()
    fs.get_file(cfg.data.name, cfg.data.name)

    df = pd.read_csv(cfg.data.name)
    inputs = torch.from_numpy(df.drop(columns=cfg.data.target).values).float()
    target = torch.from_numpy(df[cfg.data.target].values).long()

    mlflow.set_tracking_uri(cfg.mlflow.uri)
    model = mlflow.pytorch.load_model(f"models:/{cfg.model.name}/latest")

    outputs = model(inputs)

    f1 = F1Score(task=cfg.model.f1_task, num_classes=cfg.model.output_dim)
    predicted = torch.argmax(outputs, dim=1)
    test_acc = torch.sum(target == predicted).item() / (len(predicted) * 1.0)
    test_f1 = f1(predicted, target).item()

    print(f"test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}")


if __name__ == "__main__":
    infer()
