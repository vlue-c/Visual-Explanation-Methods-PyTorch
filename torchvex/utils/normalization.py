from numbers import Number
import torch


def clamp_quantile(data, q):
    if isinstance(q, Number):
        q = (0, q)

    if q != (0, 1):
        min_clip = torch.quantile(data, q[0])
        max_clip = torch.quantile(data, q[1])
        data = data.clamp(min_clip, max_clip)

    return data


def min_max_normalization(data, dim=None, q=(0, 1)):
    if dim is None:
        dim = list(range(data.ndim))
    elif isinstance(dim, Number):
        dim = (dim, )

    data = clamp_quantile(data, q)

    minima = data.clone()
    maxima = data.clone()
    for d in dim:
        minima = minima.min(d, keepdim=True)[0]
        maxima = maxima.max(d, keepdim=True)[0]

    return (data - minima) / (maxima - minima)


def denormalize(data, mean, std):
    ndim_before = data.ndim
    if ndim_before not in (3, 4):
        raise ValueError(f'data.ndim should in (3, 4). not {ndim_before}')
    if ndim_before == 3:
        data = data.unsqueeze(0)
    _, nchannels, _, _ = data.size()
    if not all(nchannels == len(static) for static in (mean, std)):
        raise ValueError('size does not match. '
                         f'channel size of data: {nchannels}, '
                         f'len(mean): {len(mean)}, '
                         f'len(std): {len(std)}.')
    mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
    std = torch.as_tensor(std, device=data.device, dtype=data.dtype)
    mean, std = mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)

    out = data * std + mean
    if ndim_before == 3:
        out = out.squeeze(0)
    return out
