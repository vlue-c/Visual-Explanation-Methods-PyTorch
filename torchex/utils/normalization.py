import torch


def min_max_normalization(data, q=1):
    if q < 1:
        max_clip = torch.quantile(data, q)
        data[data > max_clip] = max_clip

    return (data - data.min()) / (data.max() - data.min())
