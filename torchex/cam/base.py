import torch


class _CAMBase(torch.nn.Module):
    def __init__(self, model, target_layer, create_graph=False, interpolate=True):
        super().__init__()
        self.model = model
        self.target_layer = target_layer

        self.create_graph = create_graph
        self._interpolate = interpolate

    def interpolate(self, cam, shape):
        return torch.nn.functional.interpolate(
            cam, shape, mode='bilinear', align_corners=True
        )

    def create_cam(self, inputs, target):
        raise NotImplementedError

    @torch.no_grad()
    def no_grad_forward(self, inputs, target):
        return self.create_cam(inputs, target)

    @torch.enable_grad()
    def enable_grad_forward(self, inputs, target):
        return self.create_cam(inputs, target)

    @torch.no_grad()
    def _forward(self, inputs, target):
        if self.create_graph:
            res = self.enable_grad_forward(inputs, target)
        res = self.no_grad_forward(inputs, target)

        if self._interpolate:
            res = self.interpolate(res, inputs.shape[-1])

        return res

    forward = _forward
