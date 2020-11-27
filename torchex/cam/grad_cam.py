from numbers import Number

import torch

from .hook import FeatureFetcher
from .base import _CAMBase
from torchex.utils import index_to_onehot


class GradCAM(_CAMBase):
    def __init__(self, model, target_layer, create_graph=False, interpolate=True):
        if isinstance(target_layer, torch.nn.Module):
            target_layer = [target_layer]
        super().__init__(model, target_layer, create_graph, interpolate)

    @torch.enable_grad()
    def create_cam(self, inputs, target):
        self.model.zero_grad()
        inputs.requires_grad_(True)

        with FeatureFetcher(self.target_layer) as fetcher:
            output = self.model(inputs)

        onehot = index_to_onehot(target, output.shape[-1])
        loss = (output * onehot).sum()

        grad_cams = []
        for feature in fetcher.feature:
            grad = torch.autograd.grad(
                loss, feature, create_graph=self.create_graph, retain_graph=True
            )[0]
            weight = torch.nn.functional.adaptive_avg_pool2d(grad, 1)
            grad_cam = feature.mul(weight)
            grad_cam = torch.nn.functional.relu(
                grad_cam.sum(dim=1, keepdim=True))
            grad_cams.append(grad_cam)
        return grad_cams
