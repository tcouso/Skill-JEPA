from pathlib import Path

import pytorch_lightning as pl
import stable_worldmodel as swm
import torch
from torch.utils.data import DataLoader, random_split

from src.config import JEPAConfig, normalized_columns
from src.training.transforms import (
    DictCompose,
    ImagePreprocessor,
    get_column_normalizer,
)


class DataModule(pl.LightningDataModule):
    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config
        self.train_dataset: torch.utils.data.Dataset | None = None
        self.val_dataset: torch.utils.data.Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        config = self.config
        path = Path(config.evaluation.dataset_path or config.dataset.dataset_path).resolve()
        base_dataset = swm.data.load_dataset(
            str(path),
            cache_dir=str(path.parent.parent),
            num_steps=config.dataset.history_size + config.dataset.pred_horizon,
            frameskip=config.dataset.frameskip,
            keys_to_load=config.dataset.keys_to_load,
            keys_to_cache=config.dataset.keys_to_cache,
        )

        transforms = [ImagePreprocessor("pixels", "pixels", config.dataset.frame_size)]
        for col in normalized_columns(config):
            transforms.append(get_column_normalizer(base_dataset, col, col))
        base_dataset.transform = DictCompose(transforms)

        n = len(base_dataset)
        train_n = int(n * config.dataset.train_split)
        self.train_dataset, self.val_dataset = random_split(
            base_dataset,
            [train_n, n - train_n],
            generator=torch.Generator().manual_seed(config.seed),
        )

    def train_dataloader(self) -> DataLoader:
        num_workers = self.config.dataset.num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def val_dataloader(self) -> DataLoader:
        num_workers = self.config.dataset.num_workers
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
