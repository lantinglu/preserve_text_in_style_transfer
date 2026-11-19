"""Differentiable Sobel edge preservation for RGB tensors in [0, 1]."""

import torch
from torch import nn
from torch.nn import functional as F


class SobelEdgeLoss(nn.Module):
    """MSE of signed grayscale x/y gradients, with replicate padding.

    Comparing both components preserves edge strength AND direction. Kernels
    are divided by 8 so weights do not depend on an arbitrary Sobel scale.
    The target is fixed; gradients flow only through the generated image.
    """

    def __init__(self, target):
        super().__init__()
        kernel = target.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        self.register_buffer("kernels", torch.stack([kernel, kernel.T]).unsqueeze(1))
        self.register_buffer("rgb_weights", target.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.register_buffer("target_edges", self.edges(target).detach())

    def edges(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected an N x 3 x H x W RGB tensor")
        gray = (image * self.rgb_weights).sum(dim=1, keepdim=True)
        return F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), self.kernels)

    def forward(self, image):
        if image.shape[0] != self.target_edges.shape[0] or image.shape[2:] != self.target_edges.shape[2:]:
            raise ValueError("Generated image and target must have identical dimensions")
        return F.mse_loss(self.edges(image), self.target_edges)
