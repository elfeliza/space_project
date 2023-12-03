from typing import Optional

import lightning.pytorch as pl
import pandas as pd
import torch
from dvc.fs import DVCFileSystem
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target: str):
        super().__init__()
        self.X_data = torch.from_numpy(df.drop(columns=target).values).float()
        self.y_data = torch.from_numpy(df[target].values).long()

    def __len__(self):
        return self.X_data.shape[0]

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        target: str,
        val_size: float,
        seed: int,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.val_dataset = None
        self.train_dataset = None
        self.csv_path = csv_path
        self.target = target
        self.val_size = val_size
        self.seed = seed
        self.fs = DVCFileSystem()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.fs.get_file(self.csv_path, self.csv_path)

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.csv_path)
        train_df, val_df = train_test_split(
            df, test_size=self.val_size, random_state=self.seed
        )
        self.train_dataset = MyDataset(df=train_df, target=self.target)
        self.val_dataset = MyDataset(df=val_df, target=self.target)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
