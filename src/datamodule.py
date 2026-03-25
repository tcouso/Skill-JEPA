import torch
import pytorch_lightning as pl
import webdataset as wds
from torch.utils.data import DataLoader
from typing import TypedDict

from src.config import ModelConfig


class PlatonicSample(TypedDict):
    images: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor


def _process_wds_dict(sample) -> PlatonicSample:
    frame_keys = sorted(
        [k for k in sample.keys() if "frame_" in k and k.endswith(".jpg")]
    )
    images = torch.stack([sample[k] for k in frame_keys])

    return {
        "images": images,
        "actions": torch.from_numpy(sample["actions.npy"]),
        "states": torch.from_numpy(sample["states.npy"]),
    }


class PlatonicDataModule(pl.LightningDataModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.train_urls = config.train_urls
        self.val_urls = config.val_urls
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.wds_shard_shuffle_size = config.wds_shard_shuffle_size
        self.wds_sample_shuffle_size = config.wds_sample_shuffle_size

    def _build_pipeline(self, urls: str, is_train: bool) -> wds.DataPipeline:
        pipeline = [
            wds.SimpleShardList(urls),
        ]

        if is_train:
            pipeline.append(wds.shuffle(self.wds_shard_shuffle_size))

        pipeline.extend(
            [
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode("torchrgb"),
                wds.map(_process_wds_dict),
            ]
        )

        if is_train:
            pipeline.append(wds.shuffle(self.wds_sample_shuffle_size))

        pipeline.append(wds.batched(self.batch_size, partial=False))

        return wds.DataPipeline(*pipeline)

    def setup(self, stage: str = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        dataset = self._build_pipeline(self.train_urls, is_train=True)
        return wds.WebLoader(
            dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self._build_pipeline(self.val_urls, is_train=False)
        return wds.WebLoader(
            dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True
        )
