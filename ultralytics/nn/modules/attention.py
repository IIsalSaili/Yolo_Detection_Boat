# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Attention modules for the YOLO-SEA detection framework.

This module implements the SESA attention block described in YOLO-SEA (Entropy 2025, 27, 667):

- ``SENetV2``: aggregated squeeze-and-excitation channel recalibration.
- ``SimAM``:   parameter-free spatial attention (no extra learnable parameters).
- ``SESA``:    sequential fusion — SENetV2 (channels) first, then SimAM (spatial).

All three modules are channel-preserving (``c_out == c_in``), so they can be inserted anywhere in
the backbone or neck without altering the surrounding channel layout. SESA is intended to be placed
after the last ``C2f`` block of the backbone and before ``SPPF``.

Examples:
    >>> import torch
    >>> from ultralytics.nn.modules import SESA
    >>> x = torch.randn(1, 256, 20, 20)
    >>> y = SESA(256)(x)
    >>> y.shape
    torch.Size([1, 256, 20, 20])
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("SENetV2", "SimAM", "SESA")


class SENetV2(nn.Module):
    """SENetV2 — Squeeze-aggregated Excitation channel recalibration.

    Extends classic Squeeze-and-Excitation with multiple parallel ("aggregated") squeeze branches
    whose representations are concatenated before excitation, capturing richer channel-wise and
    global statistics. The block recalibrates channels via a learned multiplicative gate and keeps
    the number of channels unchanged.

    Reference: SENetV2 — Aggregated Dense Layer for Channelwise and Global Representations.
    """

    def __init__(self, c1: int, reduction: int = 16, cardinality: int = 4):
        """Initialize SENetV2.

        Args:
            c1 (int): Number of input (and output) channels.
            reduction (int): Channel reduction ratio for the squeeze bottleneck.
            cardinality (int): Number of parallel squeeze branches to aggregate.
        """
        super().__init__()
        self.c1 = c1
        d = max(c1 // reduction, 8)  # bottleneck dim per branch
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Linear(c1, d, bias=False), nn.ReLU(inplace=True)) for _ in range(cardinality)
        )
        self.excitation = nn.Sequential(nn.Linear(d * cardinality, c1, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply aggregated squeeze-and-excitation channel recalibration."""
        b, c, _, _ = x.shape
        y = self.avgpool(x).view(b, c)  # squeeze: global average pooling
        y = torch.cat([branch(y) for branch in self.branches], dim=1)  # aggregate parallel squeezes
        y = self.excitation(y).view(b, c, 1, 1)  # excitation -> channel gate
        return x * y.expand_as(x)  # recalibrate channels


class SimAM(nn.Module):
    """SimAM — parameter-free spatial attention module.

    Computes a per-neuron energy from the local self-similarity within each channel and uses its
    inverse (via a sigmoid) to weight the activations. Introduces **zero** learnable parameters.

    Energy: e*_t = 4(σ² + λ) / ((t − μ̂)² + 2σ² + 2λ), with X̃ = sigmoid(1/E) · X.

    Reference: SimAM — A Simple, Parameter-Free Attention Module for Convolutional Neural Networks.
    """

    def __init__(self, c1: int | None = None, e_lambda: float = 1e-4):
        """Initialize SimAM.

        Args:
            c1 (int, optional): Unused (kept for a uniform constructor signature with other attention
                modules so the model parser can pass channel counts positionally).
            e_lambda (float): Regularization coefficient λ in the energy function.
        """
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-free spatial attention."""
        b, c, h, w = x.size()
        n = w * h - 1  # number of other neurons in the channel
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


class SESA(nn.Module):
    """SESA — SENetV2 (channel) followed by SimAM (spatial) attention.

    Channel-preserving fused attention block from YOLO-SEA. SENetV2 recalibrates channels first, then
    SimAM applies parameter-free spatial attention. Designed to sit after the last backbone ``C2f``
    block and before ``SPPF``.
    """

    def __init__(self, c1: int, reduction: int = 16, cardinality: int = 4, e_lambda: float = 1e-4):
        """Initialize SESA.

        Args:
            c1 (int): Number of input (and output) channels.
            reduction (int): Channel reduction ratio for SENetV2.
            cardinality (int): Number of parallel squeeze branches in SENetV2.
            e_lambda (float): Regularization coefficient λ for SimAM.
        """
        super().__init__()
        self.se = SENetV2(c1, reduction, cardinality)
        self.simam = SimAM(c1, e_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SENetV2 channel attention then SimAM spatial attention."""
        return self.simam(self.se(x))
