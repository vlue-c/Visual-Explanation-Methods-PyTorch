import torch


def auc(data):
    if data.ndim == 1:
        data = data.unsqueeze(0)
    x = torch.linspace(0, 1, data.size(-1)).expand(data.size(0), -1)
    return torch.trapz(data, x)
