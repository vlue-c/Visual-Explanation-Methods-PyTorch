import torch


def index_to_onehot(indices, num_classes, dtype=torch.float):
    onehot = torch.zeros(
        indices.shape[0], num_classes, *indices.shape[1:],
        dtype=dtype, device=indices.device
    )
    onehot.scatter_(1, indices.unsqueeze(1), 1)
    return onehot


def logit_to_onehot(logits):
    num_classes = logits.size(-1)
    _, indices = logits.max(1)
