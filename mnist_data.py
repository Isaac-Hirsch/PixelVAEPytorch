from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset


_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def _read_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        magic, n_images, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST image magic number in {path}: {magic}")
        data = handle.read()
    tensor = torch.frombuffer(memoryview(data), dtype=torch.uint8).clone()
    return tensor.view(n_images, 1, rows, cols).float() / 255.0


def _read_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        magic, n_labels = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label magic number in {path}: {magic}")
        data = handle.read()
    tensor = torch.frombuffer(memoryview(data), dtype=torch.uint8).clone()
    return tensor.view(n_labels).long()


class MNISTTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.images.size(0))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def load_mnist_dataset(root: str | Path, train: bool = True, download: bool = True) -> MNISTTensorDataset:
    root = Path(root)
    raw_dir = root / "raw"
    image_key = "train_images" if train else "test_images"
    label_key = "train_labels" if train else "test_labels"
    image_path = raw_dir / _FILES[image_key]
    label_path = raw_dir / _FILES[label_key]

    if download:
        if not image_path.exists():
            _download(_BASE_URL + _FILES[image_key], image_path)
        if not label_path.exists():
            _download(_BASE_URL + _FILES[label_key], label_path)

    if not image_path.exists() or not label_path.exists():
        missing = [str(path) for path in (image_path, label_path) if not path.exists()]
        raise FileNotFoundError(
            "MNIST data files are missing. "
            f"Expected: {', '.join(missing)}. "
            "Re-run without `--no-download` or place the IDX gzip files under the data root."
        )

    return MNISTTensorDataset(_read_images(image_path), _read_labels(label_path))
