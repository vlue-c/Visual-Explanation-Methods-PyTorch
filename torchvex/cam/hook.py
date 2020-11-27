import torch


def default_transform(feature):
    if isinstance(feature, torch.Tensor):
        return feature
    return feature[0]


class FeatureFetcher():
    def __init__(self, target_layer, target='output', feature_transform=None):
        if target not in ['input', 'output']:
            raise ValueError(f'argument target=`{target}` is not valid'
                             'choice within "input" or "output".')
        self.target_layer = target_layer
        self.target = target

        self.feature = None
        self.handler = None
        self._multiple = not isinstance(target_layer, torch.nn.Module)
        if self._multiple:
            self.feature = []
            self.handler = []

        self.transform = feature_transform
        if self.transform is None:
            self.transform = default_transform

    def _fetch_output(self, module, inputs, output):
        if self._multiple:
            self.feature.append(output)
        else:
            self.feature = output

    def _fetch_input(self, module, inputs, output):
        if self._multiple:
            self.feature.append(inputs)
        else:
            self.feature = inputs

    def register(self):
        fetch = self._fetch_input
        if self.target == 'output':
            fetch = self._fetch_output

        if self._multiple:
            for layer in self.target_layer:
                self.handler.append(layer.register_forward_hook(fetch))
        else:
            self.handler = self.target_layer.register_forward_hook(fetch)

    def remove(self):
        if self._multiple:
            for handler in self.handler:
                handler.remove()
            self.handler = []
            return

        if self.handler is not None:
            self.handler.remove()
            self.handler = None

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exec_type, exc_val, exc_tb):
        if self._multiple:
            self.feature = [self.transform(feature)
                            for feature in self.feature]
        else:
            self.feature = self.transform(self.feature)
        self.remove()
