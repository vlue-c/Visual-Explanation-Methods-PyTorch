import torch


class ExplanationMethod(torch.nn.Module):
    def __init__(self, preprocess, postprocess):
        super().__init__()
        self.preprocess = preprocess
        self.postprocess = postprocess

    def process(self, inputs, target):
        raise NotImplementedError

    def _forward(self, inputs, target=None):
        out = inputs
        if self.preprocess is not None:
            out = self.preprocess(inputs)
        out = self.process(out, target)
        if self.postprocess is not None:
            out = self.postprocess(out)
        return out

    forward = _forward
