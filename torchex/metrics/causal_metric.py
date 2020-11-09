from numbers import Number
import math
import bisect

from scipy.ndimage.filters import gaussian_filter

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


def get_gaussian_filter(size, sig):
    if isinstance(size, Number):
        size = (size, size)
    kernel = torch.zeros(*size)
    size = torch.tensor(size).float()
    centroid_min = ((size-1) // 2).long()
    centroid_max = (size // 2).long()

    kernel[centroid_min[0]:centroid_max[0]+1,
           centroid_min[1]:centroid_max[1]+1].fill_(1.)

    kernel = gaussian_filter(kernel.numpy(), sig)

    return torch.tensor(kernel)


class BlurDistoration(torch.nn.Module):
    def __init__(self, size, sigma):
        super().__init__()
        self.kernel = get_gaussian_filter(size, sigma)
        self.kernel = self.kernel.expand(3, 1, *self.kernel.shape)

    @torch.no_grad()
    def _forward(self, inputs):
        return torch.nn.functional.conv2d(
            inputs, self.kernel.to(inputs.device).type(inputs.type()),
            groups=inputs.shape[1], padding=self.kernel.shape[-1]//2)

    forward = _forward


class InsertDelete(Dataset):
    def __init__(
        self, image, mask, pixel_size, distort_function,
        descending=False, insert=True
    ):
        if image.ndim != 3:
            raise ValueError(
                f'incompatible number of dimentions `image`{image.ndim}. '
                'should be 3.'
            )
        self.image = image.clone().detach()
        self.mask = mask
        self.indices = torch.argsort(
            self.mask.float().reshape(-1), descending=descending
        )
        self.distort_function = distort_function

        self.pixel_size = pixel_size
        self.num_channels = self.image.size(0)
        self.num_pix = torch.prod(
            torch.as_tensor(self.mask.shape)
        )
        self.quantile_mode = pixel_size < 1

        if insert:
            self.initial = distort_function(image)
            self.final = image
        else:
            self.initial = image
            self.final = distort_function(image)

    def __len__(self):
        if self.quantile_mode:
            return math.ceil(1. / self.pixel_size) + 1
        return self.mask.view(-1).size(0) // self.pixel_size + 1

    @torch.no_grad()
    def __getitem__(self, idx):
        if self.quantile_mode:
            region = int(self.num_pix * self.pixel_size) * idx
        else:
            region = idx * self.pixel_size
        indcs = self.indices[:region]

        current = self.initial.clone().detach().view(self.num_channels, -1)
        current[:, indcs] = self.final.view(
            self.num_channels, -1)[:, indcs].clone().detach()

        return current.view_as(self.image)


class _IDGame(torch.nn.Module):
    def __init__(self, model, pixel_size, distort_function,
                 batch_size=-1):
        super().__init__()
        self.pixel_size = pixel_size
        self.batch_size = batch_size
        self.distort_fn = distort_function
        self.model = model

    @torch.no_grad()
    def forward(self, inputs, explanations, target=None):
        if inputs.ndim == 2:
            inputs = inputs[None]
        if inputs.ndim == 3:
            inputs = inputs[None]
        if inputs.ndim != 4:
            raise ValueError(
                f'incompatible dimention of `inputs` {inputs.ndim}'
            )
        if inputs.size(0) != explanations.size(0):
            raise ValueError(
                f'size(0) of input ({inputs.size(0)}) and'
                f'size(0) of explanations ({explanations.size(0)}) is not same.'
            )

        if target is None:
            target = self.model(inputs).max(1)[1]

        generators = []
        for x, e in zip(inputs, explanations):
            generators.append(
                InsertDelete(x, e, self.pixel_size, self.distort_fn)
            )
        generator = ConcatDataset(generators)
        batch_size = self.batch_size
        if batch_size == -1:
            batch_size = len(generator)

        loader = DataLoader(generator, batch_size=batch_size)
        sizes = generator.cummulative_sizes.copy()
        target_indicator = 0
        cumulator = 0

        confidences = []
        for current in loader:
            pred = self.model(current)
            probs = pred.softmax(1)
            cumulator += current.size(0)

            if cumulator > sizes[0]:
                full_size = sizes.pop(0)
                exceeded = cumulator - full_size
                probs_remains = probs[:exceeded]
                confidence = probs_remains[
                    torch.arange(len(probs_remains)).unsqueeze(1),
                    target[[target_indicator]].unsqueeze(1)]
                confidences.append(confidence.detach().squeeze())
                probs = probs[exceeded:]
                target_indicator += 1
                cumulator = 0

            confidence = probs[
                torch.arange(len(probs)).unsqueeze(1),
                target[[target_indicator]].unsqueeze(1)]
            confidences.append(confidence.detach().squeeze())

        slicer = [len(gen) for gen in generator.datasets]
        confidences = torch.cat(confidences)
        splited = confidences.split(slicer)
        return torch.stack(splited)
