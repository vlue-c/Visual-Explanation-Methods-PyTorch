from numbers import Number
import torch

from .base import _CAMBase
from .hook import FeatureFetcher


class CAM(_CAMBase):
    def __init__(self, model, target_layer, fc_layer, create_graph=False,
                 interpolate=True, preprocess=None, postprocess=None):
        super().__init__(model, target_layer, create_graph, interpolate,
                         preprocess, postprocess)
        self.fc = fc_layer

    def create_cam(self, inputs, target):
        with FeatureFetcher(self.target_layer) as fetcher:
            _ = self.model(inputs)

        weight = self.fc.weight

        cam = torch.tensordot(weight, fetcher.feature, [[1], [1]])
        cam = cam.permute(1, 0, 2, 3)

        batch_selector = torch.arange(len(target)).unsqueeze(1)
        cam = cam[batch_selector, target.unsqueeze(1)]
        return cam
