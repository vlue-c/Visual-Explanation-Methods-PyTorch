import random
import torch
from torch.nn.functional import interpolate
from torchvision.transforms.functional import gaussian_blur


def gaussian_filter(image, sigma=10, truncate=4.0):
    radius = int(truncate * sigma + 0.5)
    radius += (radius-1) % 2
    return gaussian_blur(image, radius, sigma)


def make_blurred_circular_mask(image, rads=torch.arange(0, 175, 5)):
    h, w = image.shape[-2:]
    y_center, x_center = h // 2, w // 2

    rads = rads.view(rads.size(0), 1, 1, 1).to(image.device)

    y = torch.arange(-y_center, h-y_center,
                     dtype=image.dtype, device=image.device)[..., None]
    x = torch.arange(-x_center, w-x_center,
                     dtype=image.dtype, device=image.device)[None, ...]

    y = y.view(1, 1, -1, 1).expand(rads.size(0), -1, -1, -1)
    x = x.view(1, 1, 1, -1).expand(rads.size(0), -1, -1, -1)

    mask = (y.pow(2) + x.pow(2)) < rads.pow(2)
    mask = mask.float()

    return mask


def total_variation(inputs, beta=1):
    dh = inputs[..., 1:, :] - inputs[..., :-1, :]
    dw = inputs[..., :, 1:] - inputs[..., :, :-1]

    return dh.abs().pow(beta).sum() + dw.abs().pow(beta).sum()


class MeaningfulPerturbation(torch.nn.Module):
    def __init__(self, model, num_iters=300, lr=1e-1, l1_lambda=1e-4, jitter=4,
                 tv_beta=3, tv_lambda=1e-2, mask_scale=8, noise=0,
                 blur_mask_sigma=5, blur_image_sigma=10):
        super().__init__()
        self.model = model
        self.niters = num_iters
        self.lr = lr
        self.l1_l = l1_lambda
        self.tv_l = tv_lambda
        self.tv_b = tv_beta
        self.mask_scale = mask_scale
        self.blur_sigma = blur_mask_sigma
        self.jitter = jitter
        self.sd = blur_image_sigma
        self.noise = noise

    def _initialize_mask(self, image, blurred_image, target):
        original_score = self.model(image)[0, target]
        masks = make_blurred_circular_mask(image)
        masks = gaussian_filter(masks, self.sd)
        scores = self.model(image * (1-masks) + blurred_image * masks)
        scores = scores.softmax(1)
        scores = scores[
            torch.arange(scores.size(0)).unsqueeze(1),
            target.unsqueeze(1)
        ].squeeze()
        percs = (scores - scores[-1]) / (original_score - scores[-1])
        first_i = torch.nonzero(percs < 1e-2)

        if first_i.size(0) == 0:
            initial_mask = masks[[-1]]
        else:
            initial_mask = masks[first_i[0, 0][None]]

        if self.mask_scale > 0:
            initial_mask = interpolate(
                initial_mask, scale_factor=1 / self.mask_scale,
                mode='nearest'
            )
        initial_mask.clamp_(0, 1)
        return 1 - initial_mask

    def _forward(self, inputs, blurred_inputs, target=None):
        if target is None:
            target = self.model(inputs).argmax(1)

        mask = self._initialize_mask(inputs, blurred_inputs, target)
        j = self.jitter
        if j > 0:
            h, w = inputs.shape[-2:]
            inputs = interpolate(inputs, (h+j, w+j),
                                 mode='bilinear', align_corners=True)
            blurred_inputs = interpolate(blurred_inputs, (h+j, w+j),
                                         mode='bilinear', align_corners=True)

        mask.requires_grad_(True)
        mask = torch.nn.Parameter(mask)

        optimizer = torch.optim.Adam([mask], lr=self.lr)

        for _ in range(self.niters):
            optimizer.zero_grad()
            j_h = j_w = 0
            if j > 0:
                j_h = random.randrange(0, j)
                j_w = random.randrange(0, j)

            img = inputs.clone().detach()[
                ..., j_h:j_h+inputs.size(-2)-j, j_w:j_w+inputs.size(-1)-j
            ]
            bimg = blurred_inputs.clone().detach()[
                ..., j_h:j_h+inputs.size(-2)-j, j_w:j_w+inputs.size(-1)-j
            ]

            noise = 0
            if self.noise > 0:
                noise = torch.normal(0, self.noise, size=mask.shape)

            mask_inter = mask + noise
            mask_inter = mask_inter.clamp(0, 1)

            if self.mask_scale > 0:
                mask_inter = interpolate(
                    mask_inter, scale_factor=self.mask_scale, mode='nearest'
                )
            if self.blur_sigma > 0:
                mask_inter = gaussian_filter(mask_inter, self.blur_sigma)

            masked_img = img * mask_inter + bimg * (1 - mask_inter)
            out = self.model(masked_img).softmax(1)
            target_score = out[0, target]

            l1_loss = self.l1_l * (mask - 1).abs().sum()
            tv_loss = self.tv_l * total_variation(mask, self.tv_b)

            loss = target_score + l1_loss + tv_loss

            loss.backward()
            optimizer.step()

            mask.data.clamp_(0, 1)

        if self.mask_scale > 0:
            mask = interpolate(
                mask, scale_factor=self.mask_scale, mode='nearest'
            )
        if self.blur_sigma > 0:
            mask = gaussian_filter(mask, self.blur_sigma)
        return mask.data

    forward = _forward
