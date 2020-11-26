import torch

from torchex.base import ExplanationMethod
from torchex import SimpleGradient


class SmoothGradient(ExplanationMethod):
    def __init__(self, model, stdev_spread=0.15, num_samples=25,
                 magnitude=True, batch_size=-1,
                 create_graph=False, preprocess=None, postprocess=None):
        super().__init__(preprocess, postprocess)
        self.model = model
        self.stdev_spread = stdev_spread
        self.nsample = num_samples
        self.create_graph = create_graph
        self.magnitude = magnitude
        self.batch_size = batch_size
        if self.batch_size == -1:
            self.batch_size = self.nsample

        self._simgrad_gen = SimpleGradient(model, create_graph)

    def process(self, inputs, target=None):
        if target is None:
            target = self.model(inputs).argmax(1)

        self.model.zero_grad()

        maxima = inputs.flatten(1).max(-1)[0]
        minima = inputs.flatten(1).min(-1)[0]

        stdev = self.stdev_spread * (maxima - minima).cpu()
        stdev = stdev.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        stdev = stdev.unsqueeze(0).repeat(self.nsample, 1, 1, 1, 1)
        noise = torch.normal(0, stdev)

        noiseloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(noise), batch_size=self.batch_size
        )

        total_gradients = torch.zeros_like(inputs)
        for noise in noiseloader:
            inputs_w_noise = inputs.unsqueeze(0) + noise[0].to(inputs.device)
            inputs_w_noise = inputs_w_noise.view(-1, *inputs.shape[1:])
            gradients = self._simgrad_gen(inputs_w_noise)
            gradients = gradients.view(self.batch_size, *inputs.shape)
            if self.magnitude:
                gradients = gradients.pow(2)
            total_gradients = total_gradients + gradients.sum(0)

        smoothed_gradient = total_gradients / self.nsample
        return smoothed_gradient
