from numbers import Number
import torch


def min_max_normalization(data, dim=None, q=(0, 1)):
    if isinstance(q, Number):
        q = (0, q)
    if dim is None:
        dim = list(range(data.ndim))
    if isinstance(dim, Number):
        dim = (dim, )

    if q != (0, 1):
        min_clip = torch.quantile(data, q[0])
        max_clip = torch.quantile(data, q[1])
        data = data.clamp(min_clip, max_clip)

    minima = data.clone()
    maxima = data.clone()
    for d in dim:
        minima = minima.min(d, keepdim=True)[0]
        maxima = maxima.max(d, keepdim=True)[0]

    return (data - minima) / (maxima - minima)


def denormalize(data, mean, std):
    mean = torch.tensor(mean).expand_as(data)
    std = torch.tensor(std).expand_as(data)
    return data * std + mean
