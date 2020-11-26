import os
import torch
from torch.utils.data import Subset
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
