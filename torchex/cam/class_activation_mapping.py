from numbers import Number
import torch

from .base import _CAMBase
from .hook import FeatureFetcher


class CAM(_CAMBase):
    def __init__(self, model, target_layer, fc_layer=None, create_graph=False, interpolate=True):
        super().__init__(model, target_layer, create_graph, interpolate)
        self.fc = fc_layer
        if fc_layer is None:
            self.fc = fc_layer

    def create_cam(self, inputs, target=None):
        with FeatureFetcher(self.target_layer) as fetcher:
            output = self.model(inputs)

        if target is None:
            _, target = output.max(1)
        if isinstance(target, Number):
            target = [target]
        target = torch.tensor(target)

        weight = self.fc.weight

        cam = torch.tensordot(weight, fetcher.feature, [[1], [1]])
        cam = cam.permute(1, 0, 2, 3)

        batch_selector = torch.arange(len(target)).unsqueeze(1)
        cam = cam[batch_selector, target.unsqueeze(1)]
        return cam
