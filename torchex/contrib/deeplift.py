import torch
from captum.attr import DeepLift as _DeepLift

from torchex.base import ExplanationMethod


class DeepLift(ExplanationMethod):
    def __init__(self, model, baseline=None, preprocess=None, postprocess=None):
        super().__init__(preprocess, postprocess)
        self.model = model
        self.attributor = _DeepLift(model)
        self.baseline = baseline

    @torch.enable_grad()
    def process(self, inputs, target):
        baseline = self.baseline
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        attribution = self.attributor.attribute(inputs, baseline, target)
        return attribution
