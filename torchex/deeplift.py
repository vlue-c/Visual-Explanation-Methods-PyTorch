import torch
from captum.attr import DeepLift as _DeepLift


class DeepLift(torch.nn.Module):
    def __init__(self, model, baseline=None, post_process=None):
        super().__init__()
        self.model = model
        self.attributor = _DeepLift(model)
        self.baseline = baseline
        self.post_process = post_process

    @torch.enable_grad()
    def _forward(self, inputs, target):
        baseline = self.baseline
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        attribution = self.attributor.attribute(inputs, baseline, target)
        if self.post_process is not None:
            attribution = self.post_process(attribution)
        return attribution

    forward = _forward
