import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional

import stable_worldmodel as swm
import stable_pretraining as spt
from utils import get_column_normalizer, get_img_preprocessor
from src.config import ModelConfig


class LeWMDataModule(pl.LightningDataModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.dataset_name = config.dataset.name
        self.frameskip = config.dataset.frameskip
        self.history_size = config.dataset.history_size
        self.num_preds = config.action.sequence_length
        self.batch_size = config.training.batch_size
        self.num_workers = config.dataset.num_workers
        self.train_split = config.dataset.train_split
        self.seed = config.seed
        self.img_size = config.vision.frame_size

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        total_steps = self.history_size + self.num_preds

        base_dataset = swm.data.HDF5Dataset(
            name=self.dataset_name,
            num_steps=total_steps,
            frameskip=self.frameskip,
            keys_to_load=["pixels", "action", "proprio"],
            keys_to_cache=["action", "proprio"],
            transform=None
        )

        transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=self.img_size)]

        for col in ["action", "proprio"]:
            normalizer = get_column_normalizer(base_dataset, col, col)
            transforms.append(normalizer)
            setattr(self.config, f"{col}_dim", base_dataset.get_dim(col))

        base_dataset.transform = spt.data.transforms.Compose(*transforms)

        rnd_gen = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = spt.data.random_split(
            base_dataset, 
            lengths=[self.train_split, 1.0 - self.train_split], 
            generator=rnd_gen
        )

    def train_dataloader(self) -> DataLoader:
        rnd_gen = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            generator=rnd_gen
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )