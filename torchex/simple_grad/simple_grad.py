import torch


class SimpleGradient(torch.nn.Module):
    def __init__(self, model, create_graph=False, post_process=None):
        super().__init__()
        self.model = model
        self.create_graph = create_graph
        self.post_process = post_process

    def predict(self, x):
        return self.model(x)

    @torch.enable_grad()
    def _forward(self, inputs, target=None):
        self.model.zero_grad()
        inputs.requires_grad_(True)

        out = self.model(inputs)
        if target is None:
            target = out.max(1)[1]

        num_classes = out.size(-1)
        onehot = torch.zeros(inputs.size(0), num_classes, *target.shape[1:])
        onehot = onehot.to(dtype=inputs.dtype, device=inputs.device)
        onehot.scatter_(1, target.unsqueeze(1), 1)

        grad, = torch.autograd.grad(
            (out*onehot).sum(), inputs, create_graph=self.create_graph
        )

        if self.post_process is not None:
            grad = self.post_process(grad)
        return grad

    forward = _forward
