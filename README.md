# PyTorch-Explanations

## Re-implemented model explanation method codes for PyTorch.

- PyTorch friendly
- More utilization of GPU.
- Higher-order derivative friendly
- Batch processing

## Usage

```sh
pip install -r requirements.txt
python setup.py install
```

## requirements

- pytorch >= 1.7 (finally torch.quantile)
- tqdm
- ***TBD***

optional
- [captum](https://captum.ai/) (for now. for DeepLift)

---

## Class Activation Mapping based Methods
- **C**lass **A**ctivation **M**apping(*CAM*)
  - paper (*CVPR* 2016): [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)
  - original source code: https://github.com/zhoubolei/CAM
- Funtionality:

| | |
| --- | --- |
| `Higher order derivative` | :heavy_check_mark: |
| `Batch processing` | :heavy_check_mark: |
| `post processing` | :x: |

Expample: 
```python
from torchvision.models import resnet50
from torchex import CAM

# in case of torchvision.models.resnet,
# the output of the model.layer4 is equals to
# the output of the last conv-bn-relu layer
resnet = resnet50(pretrained=True).eval()
cam_generator = CAM(resnet, target_layer=model.layer4, fc_layer=model.fc)

# if no target is passed,
# the predicted class is used as the target.
cam = cam_generator(image)
cam = cam_generator(image, target)
```

- **Grad**ient-weighted **C**lass **A**ctivation **M**apping(*Grad-CAM*)
  - paper (*ICCV* 2017): [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)
  - original source code: ***TBD***

Example:
```python
from torchex import GradCAM
model = ...
grad_cam_generator = GradCAM(model, target_layer=model.layer4)
multil_layer_gcamgen = GradCAM(model, target_layer=[model.layer3, model.layer4])

grad_cam = grad_cam_generator(image)
grad_cam = grad_cam_generator(image, target)

multiple_grad_cam = multil_layer_gcamgen(image)
# multiple_grad_cam.shape: torch.Size([2, 1, image.size(-2), image.size(-1)])
```

## Simple Gradient
- ![](https://latex.codecogs.com/svg.latex?SimpleGradient=\frac{\partial%20M(x)_c}{\partial%20x})

- where *M* is model, *c* is target class, *x* is input image.

Example:
```python
from torchex import SimpleGradient

def normalize_gradient(gradient):
  nbatchs, nchannels, w, h = gradient.shape
  return w * h * gradient / gradient.sum()

model = ...
simgrad_generator = SimpleGradient(model, post_process=normalize_gradient)

simgrad = simgrad_generator(image)
simgrad = simgrad_generator(image, target)
```

## DeepLift

- **Deep** **L**earning **I**mportant **F**ea**T**ures
  - ***For now, it is just a wrapper class of the  [`captum`](https://captum.ai/)`.attr.DeepLift`***
  - paper (*arXiv preprint* 2017):
  [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685)
  - original source code: https://github.com/kundajelab/deeplift

## RISE

- **R**andomized **I**mage **S**ampling for **E**xplanations
  - paper (*BMVC*, 2018): [RISE: Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/abs/1806.07421)
  - original source code: https://github.com/eclique/RISE

Example:
```python
from torchex import RISE

model = ...
rise_generator = RISE(model, num_masks=8000, cell_size=7,
                      probability=0.5, batch_size=256)

rise = rise_generator(image)
rise = rise_generator(image, target)
```