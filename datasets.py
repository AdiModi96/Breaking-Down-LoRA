import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import torchvision.datasets
from torchvision import transforms

import project_paths as pp

transforms = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)


class MNIST(Dataset):
    def __init__(self, train: bool = False) -> None:
        self.dataset = torchvision.datasets.MNIST(
            root=pp.datasets_folder_path,
            train=train,
            download=True,
            transform=transforms
        )

        self.num_records = len(self.dataset)

    def __len__(self) -> int:
        return self.num_records

    def __getitem__(self, idx: int) -> (Tensor, Tensor):
        image, label = self.dataset[idx]

        target = torch.zeros(size=(10,))
        target[label] = 1

        return image, target


class FashionMNIST(Dataset):
    LABEL_MAP = {
        0: 'T-Shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot'
    }

    def __init__(self, train: bool = False) -> None:
        self.dataset = torchvision.datasets.FashionMNIST(
            root=pp.datasets_folder_path,
            train=train,
            download=True,
            transform=transforms
        )

        self.num_records = len(self.dataset)

    def __len__(self) -> int:
        return self.num_records

    def __getitem__(self, idx: int) -> (Tensor, Tensor):
        image, label = self.dataset[idx]

        target = torch.zeros(size=(10,))
        target[label] = 1

        return image, target
