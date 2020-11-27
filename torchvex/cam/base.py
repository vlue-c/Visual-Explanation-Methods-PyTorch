import torch

from torchvex.base import ExplanationMethod


class _CAMBase(ExplanationMethod):
    def __init__(self, model, target_layer, create_graph=False,
                 interpolate=True, preprocess=None, postprocess=None):
        super().__init__(model, preprocess, postprocess)
        self.target_layer = target_layer

        self.create_graph = create_graph
        self._interpolate = interpolate

    def interpolate(self, cam, shape):
        if cam.ndim == 3:
            cam.unsqueeze_(0)
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
    def process(self, inputs, target):
        if self.create_graph:
            results = self.enable_grad_forward(inputs, target)
        else:
            results = self.no_grad_forward(inputs, target)

        if self._interpolate:
            results = [self.interpolate(result, inputs.shape[-1])
                       for result in results]

        if all(results[0].shape == result.shape for result in results):
            return torch.cat(results)
        return results
