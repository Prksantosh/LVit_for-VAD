# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 16:02:32 2026

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. SSIM Loss
# ============================================================

class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) loss.

    Expected input:
        prediction : [B, C, H, W]
        target     : [B, C, H, W]

    Assumes pixel values are normalized to [0, 1].

    Loss:
        L_ssim = 1 - SSIM(prediction, target)
    """

    def __init__(
        self,
        window_size=11,
        sigma=1.5,
        data_range=1.0,
        eps=1e-8
    ):
        super().__init__()

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.eps = eps

        # SSIM constants
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def _gaussian_kernel(
        self,
        channels,
        device,
        dtype
    ):
        """
        Construct a 2D Gaussian window.
        """

        coords = torch.arange(
            self.window_size,
            device=device,
            dtype=dtype
        )

        coords = coords - (
            self.window_size - 1
        ) / 2.0

        gaussian_1d = torch.exp(
            -(coords ** 2)
            / (2 * self.sigma ** 2)
        )

        gaussian_1d = (
            gaussian_1d
            / gaussian_1d.sum()
        )

        gaussian_2d = (
            gaussian_1d[:, None]
            @ gaussian_1d[None, :]
        )

        kernel = gaussian_2d.expand(
            channels,
            1,
            self.window_size,
            self.window_size
        ).contiguous()

        return kernel

    def forward(
        self,
        prediction,
        target
    ):
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} "
                f"does not match target shape {target.shape}"
            )

        channels = prediction.shape[1]

        kernel = self._gaussian_kernel(
            channels=channels,
            device=prediction.device,
            dtype=prediction.dtype
        )

        padding = self.window_size // 2

        # ------------------------------------
        # Local means
        # ------------------------------------

        mu_x = F.conv2d(
            prediction,
            kernel,
            padding=padding,
            groups=channels
        )

        mu_y = F.conv2d(
            target,
            kernel,
            padding=padding,
            groups=channels
        )

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        mu_xy = mu_x * mu_y

        # ------------------------------------
        # Local variances and covariance
        # ------------------------------------

        sigma_x_sq = (
            F.conv2d(
                prediction * prediction,
                kernel,
                padding=padding,
                groups=channels
            )
            - mu_x_sq
        )

        sigma_y_sq = (
            F.conv2d(
                target * target,
                kernel,
                padding=padding,
                groups=channels
            )
            - mu_y_sq
        )

        sigma_xy = (
            F.conv2d(
                prediction * target,
                kernel,
                padding=padding,
                groups=channels
            )
            - mu_xy
        )

        # Numerical protection
        sigma_x_sq = torch.clamp(
            sigma_x_sq,
            min=0.0
        )

        sigma_y_sq = torch.clamp(
            sigma_y_sq,
            min=0.0
        )

        # ------------------------------------
        # SSIM
        # ------------------------------------

        numerator = (
            (2.0 * mu_xy + self.C1)
            *
            (2.0 * sigma_xy + self.C2)
        )

        denominator = (
            (mu_x_sq + mu_y_sq + self.C1)
            *
            (
                sigma_x_sq
                + sigma_y_sq
                + self.C2
            )
        )

        ssim_map = numerator / (
            denominator + self.eps
        )

        ssim_value = ssim_map.mean()

        # Convert similarity into loss
        return 1.0 - ssim_value


# ============================================================
# 2. Gradient Loss
# ============================================================

class GradientLoss(nn.Module):
    """
    Gradient/edge reconstruction loss using Sobel operators.

    Measures the difference between horizontal and vertical
    spatial gradients of reconstructed and target frames.

    Default:
        L1 difference between gradients.
    """

    def __init__(
        self,
        loss_type="l1"
    ):
        super().__init__()

        self.loss_type = loss_type.lower()

        # Sobel X
        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0]
            ],
            dtype=torch.float32
        )

        # Sobel Y
        sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [ 0.0,  0.0,  0.0],
                [ 1.0,  2.0,  1.0]
            ],
            dtype=torch.float32
        )

        self.register_buffer(
            "sobel_x",
            sobel_x.view(
                1, 1, 3, 3
            )
        )

        self.register_buffer(
            "sobel_y",
            sobel_y.view(
                1, 1, 3, 3
            )
        )

    def _compute_gradients(
        self,
        x
    ):
        """
        Compute channel-wise Sobel gradients.
        """

        channels = x.shape[1]

        kernel_x = self.sobel_x.repeat(
            channels,
            1,
            1,
            1
        )

        kernel_y = self.sobel_y.repeat(
            channels,
            1,
            1,
            1
        )

        grad_x = F.conv2d(
            x,
            kernel_x,
            padding=1,
            groups=channels
        )

        grad_y = F.conv2d(
            x,
            kernel_y,
            padding=1,
            groups=channels
        )

        return grad_x, grad_y

    def forward(
        self,
        prediction,
        target
    ):
        pred_gx, pred_gy = (
            self._compute_gradients(
                prediction
            )
        )

        target_gx, target_gy = (
            self._compute_gradients(
                target
            )
        )

        if self.loss_type == "l1":

            loss_x = F.l1_loss(
                pred_gx,
                target_gx
            )

            loss_y = F.l1_loss(
                pred_gy,
                target_gy
            )

        elif self.loss_type == "mse":

            loss_x = F.mse_loss(
                pred_gx,
                target_gx
            )

            loss_y = F.mse_loss(
                pred_gy,
                target_gy
            )

        else:

            raise ValueError(
                "loss_type must be "
                "'l1' or 'mse'"
            )

        return 0.5 * (
            loss_x + loss_y
        )


# ============================================================
# 3. Composite Reconstruction Loss
# ============================================================

class CompositeReconstructionLoss(nn.Module):
    """
    Composite objective for the proposed
    LViT + FASC + PSFE autoencoder.

    L_total =
        lambda_mse  * L_MSE
      + lambda_ssim * L_SSIM
      + lambda_grad * L_gradient

    Recommended initial setting:

        lambda_mse  = 0.5
        lambda_ssim = 1.0
        lambda_grad = 0.1

    The three terms respectively constrain:

        MSE      -> pixel fidelity
        SSIM     -> structural fidelity
        Gradient -> edges and textures
    """

    def __init__(
        self,
        lambda_mse=0.5,
        lambda_ssim=1.0,
        lambda_grad=0.1,
        ssim_window_size=11,
        ssim_sigma=1.5,
        data_range=1.0,
        gradient_loss_type="l1"
    ):
        super().__init__()

        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad

        self.mse_loss = nn.MSELoss()

        self.ssim_loss = SSIMLoss(
            window_size=ssim_window_size,
            sigma=ssim_sigma,
            data_range=data_range
        )

        self.gradient_loss = GradientLoss(
            loss_type=gradient_loss_type
        )

    def forward(
        self,
        prediction,
        target
    ):
        # ------------------------------------
        # Pixel reconstruction
        # ------------------------------------

        loss_mse = self.mse_loss(
            prediction,
            target
        )

        # ------------------------------------
        # Structural reconstruction
        # ------------------------------------

        loss_ssim = self.ssim_loss(
            prediction,
            target
        )

        # ------------------------------------
        # Edge / gradient reconstruction
        # ------------------------------------

        loss_grad = self.gradient_loss(
            prediction,
            target
        )

        # ------------------------------------
        # Weighted composite loss
        # ------------------------------------

        total_loss = (
            self.lambda_mse
            * loss_mse
            +
            self.lambda_ssim
            * loss_ssim
            +
            self.lambda_grad
            * loss_grad
        )

        # Return total + individual components
        return {
            "total_loss": total_loss,

            "mse_loss": loss_mse,

            "ssim_loss": loss_ssim,

            "gradient_loss": loss_grad
        }