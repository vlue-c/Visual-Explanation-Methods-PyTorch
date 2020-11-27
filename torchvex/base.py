from numbers import Number

import torch


class ExplanationMethod(torch.nn.Module):
    def __init__(self, model, preprocess, postprocess):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def process(self, inputs, target):
        raise NotImplementedError

    def _forward(self, inputs, target=None):
        if target is None:
            target = self.model(inputs).argmax(1)
        if isinstance(target, Number):
            target = [target]
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target)
        out = inputs
        if self.preprocess is not None:
            out = self.preprocess(inputs)

        if out.device != target.device:
            target = target.to(out.device)
        out = self.process(out, target)
        if self.postprocess is not None:
            out = self.postprocess(out)
        return out

    forward = _forward
