# PyTorch-Explanations

## Re-implemented model explanation method codes for PyTorch.

---

## Class Activation Mapping based Methods
- Class Activation Mapping(*CAM*)
  - paper (*CVPR* 2016): https://arxiv.org/pdf/1512.04150.pdf
  - original source code: https://github.com/zhoubolei/CAM

- Gradient-weighted Class Activation Mapping(*Grad-CAM*)
  - paper (*ICCV* 2017): https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
  - original source code: <u>TBD</u>

## Simple Gradient

- Simple gradient method:

  - ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20M%28x%29_%7Bc%7D%7D%7B%5Cpartial%20x%7D)
  - where *M* is model, *c* is target class, *x* is input.


