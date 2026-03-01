import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, TypedDict

class PlatonicSample(TypedDict):
    images: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor

class PlatonicSolidsDataset(Dataset):
    def __init__(self, data: Dict[str, List], transform=None):
        self.images = torch.tensor(data["images"], dtype=torch.uint8)
        self.actions = torch.tensor(data["actions"], dtype=torch.float32)
        self.states = torch.tensor(data["states"], dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> PlatonicSample:
        sample_data = {
            "images": self.images[index],
            "actions": self.actions[index],
            "states": self.states[index],
        }
        sample = PlatonicSample(**sample_data)

        if self.transform:
            sample = self.transform(sample)

        return sample


def read_platonic_solids_dataset(dataset_dir: str) -> PlatonicSolidsDataset:
    # Sort to ensure deterministic ordering of data
    shard_files = sorted(
        [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".npz") or f.endswith(".npy")
        ]
    )

    all_images = []
    all_actions = []
    all_states = []

    for file_path in shard_files:
        shard_data = np.load(file_path, allow_pickle=True)

        all_images.append(shard_data["images"])
        all_actions.append(shard_data["actions"])
        all_states.append(shard_data["states"])

    combined_data = {
        "images": np.concatenate(all_images, axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "states": np.concatenate(all_states, axis=0),
    }

    return PlatonicSolidsDataset(combined_data)
