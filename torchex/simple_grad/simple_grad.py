import torch

from torchex.base import ExplanationMethod


class SimpleGradient(ExplanationMethod):
    def __init__(self, model, create_graph=False,
                 preprocess=None, postprocess=None):
        super().__init__(model, preprocess, postprocess)
        self.create_graph = create_graph

    def predict(self, x):
        return self.model(x)

    @torch.enable_grad()
    def process(self, inputs, target):
        self.model.zero_grad()
        inputs.requires_grad_(True)

        out = self.model(inputs)

        num_classes = out.size(-1)
        onehot = torch.zeros(inputs.size(0), num_classes, *target.shape[1:])
        onehot = onehot.to(dtype=inputs.dtype, device=inputs.device)
        onehot.scatter_(1, target.unsqueeze(1), 1)

        grad, = torch.autograd.grad(
            (out*onehot).sum(), inputs, create_graph=self.create_graph
        )

        return grad
