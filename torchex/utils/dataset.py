import os
import torch
from torch.utils.data import Subset, Dataset
from torchvision import transforms as T

IMAGENET_MEAN = torch.as_tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.as_tensor([0.229, 0.224, 0.225])


def imagenet_transform(centercrop=True):
    transform = []
    if centercrop:
        transform.extend([
            T.Resize(256),
            T.CenterCrop(224),
        ])
    else:
        transform.append(T.Resize((224, 224)))
    transform.extend([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return T.Compose(transform)


def make_subset_from_fnames(dset, fnames):
    samples = dset.samples
    paths, _ = list(map(list, zip(*samples)))
    paths = [os.path.basename(path) for path in paths]
    indices = []

    for fname in fnames:
        indices.append(paths.index(os.path.basename(fname)))

    return Subset(dset, indices)


class Pairset(Dataset):
    def __init__(self, *datasets, transform=None):
        super().__init__()
        lengths = list(map(len, datasets))
        if not all(lengths[0] == length for length in lengths):
            raise ValueError(f'length of each dataset is not same {lengths}')
        self.datasets = datasets
        self.transform = transform

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        samples = [dset[idx] for dset in self.datasets]

        if self.transform is not None:
            sample = self.transform(samples)

        return sample
