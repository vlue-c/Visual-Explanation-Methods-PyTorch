# PyTorch-Explanations

## Re-implemented model explanation method codes for PyTorch.

---

## Class Activation Mapping based Methods
- **C**lass **A**ctivation **M**apping(*CAM*)
  - paper (*CVPR* 2016): [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)
  - original source code: https://github.com/zhoubolei/CAM

- **Grad**ient-weighted **C**lass **A**ctivation **M**apping(*Grad-CAM*)
  - paper (*ICCV* 2017): [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)
  - original source code: <u>TBD</u>

## Simple Gradient
- ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20M%28x%29_%7Bc%7D%7D%7B%5Cpartial%20x%7D)

- where *M* is model, *c* is target class, *x* is input.


## DeepLift

- **Deep** **L**earning **I**mportant **F**ea**T**ures
  - For now it is constructed as a wrapper class of the  [`captum`](https://captum.ai/)`.attr.DeepLift`
  - paper (*arXiv preprint* 2017):
  [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685)
  - original source code: https://github.com/kundajelab/deeplift

## RISE

- **R**andomized **I**mage **S**ampling for **E**xplanations
  - paper (*BMVC*, 2018): [RISE: Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/abs/1806.07421)
  - original source code: https://github.com/eclique/RISE