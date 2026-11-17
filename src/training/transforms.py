"""Image and column transforms, for training's dict samples and eval's plain tensors."""

import numpy as np
import torch
from torchvision.transforms import v2 as transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(img_size: int) -> transforms.Compose:
    """uint8 HWC array/tensor -> ImageNet-normalised float CHW tensor, resized to img_size."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
            transforms.Resize(img_size, antialias=True),
        ]
    )


class ImagePreprocessor:
    """Dict-keyed transform: uint8 HWC numpy array → normalised float CHW tensor."""

    def __init__(self, source: str, target: str, img_size: int):
        self._src, self._tgt = source, target
        self._t = build_image_transform(img_size)

    def __call__(self, sample: dict) -> dict:
        sample[self._tgt] = self._t(sample[self._src])
        return sample


class ColumnNormalizer:
    """Dict-keyed transform: (x - mean) / std for one column."""

    def __init__(self, source: str, target: str, mean: torch.Tensor, std: torch.Tensor):
        self._src, self._tgt = source, target
        self._mean, self._std = mean, std

    def __call__(self, sample: dict) -> dict:
        sample[self._tgt] = ((sample[self._src] - self._mean) / self._std).float()
        return sample


class DictCompose:
    """Apply a list of dict-keyed transforms in sequence."""

    def __init__(self, transforms: list):
        self._transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self._transforms:
            sample = t(sample)
        return sample


def get_column_normalizer(dataset, source: str, target: str) -> ColumnNormalizer:
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return ColumnNormalizer(source, target, mean, std)
