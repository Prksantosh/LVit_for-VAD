import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_finite(name, x):
    """Raise a useful error as soon as NaN/Inf enters the loss."""
    if not torch.isfinite(x).all():
        with torch.no_grad():
            finite = x[torch.isfinite(x)]
            if finite.numel() > 0:
                lo = finite.min().item()
                hi = finite.max().item()
                msg = f"finite range=[{lo:.6g}, {hi:.6g}]"
            else:
                msg = "tensor contains no finite values"
        raise FloatingPointError(f"Non-finite values detected in {name}; {msg}")


class SSIMLoss(nn.Module):
    """Differentiable and AMP-safe SSIM loss for [B,C,H,W] images."""

    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, eps=1e-6):
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if data_range <= 0:
            raise ValueError("data_range must be > 0")

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.eps = eps
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def _gaussian_kernel(self, channels, device, dtype):
        coords = torch.arange(self.window_size, device=device, dtype=dtype)
        coords = coords - (self.window_size - 1) / 2.0
        gaussian_1d = torch.exp(-(coords.square()) / (2.0 * self.sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum().clamp_min(self.eps)
        gaussian_2d = gaussian_1d[:, None] @ gaussian_1d[None, :]
        return gaussian_2d.expand(
            channels, 1, self.window_size, self.window_size
        ).contiguous()

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} does not match "
                f"target shape {target.shape}"
            )
        if prediction.ndim != 4:
            raise ValueError(
                f"SSIMLoss expects [B,C,H,W], got shape {prediction.shape}"
            )

        _check_finite("prediction before SSIM", prediction)
        _check_finite("target before SSIM", target)

        # Critical AMP fix: SSIM contains squares/products/reductions that are
        # unnecessarily fragile in float16/bfloat16. Compute them in FP32.
        pred = prediction.float()
        tgt = target.float()

        channels = pred.shape[1]
        kernel = self._gaussian_kernel(
            channels=channels, device=pred.device, dtype=pred.dtype
        )
        padding = self.window_size // 2

        mu_x = F.conv2d(pred, kernel, padding=padding, groups=channels)
        mu_y = F.conv2d(tgt, kernel, padding=padding, groups=channels)

        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(pred.square(), kernel, padding=padding, groups=channels)
            - mu_x_sq
        ).clamp_min(0.0)

        sigma_y_sq = (
            F.conv2d(tgt.square(), kernel, padding=padding, groups=channels)
            - mu_y_sq
        ).clamp_min(0.0)

        sigma_xy = (
            F.conv2d(pred * tgt, kernel, padding=padding, groups=channels)
            - mu_xy
        )

        numerator = (2.0 * mu_xy + self.C1) * (2.0 * sigma_xy + self.C2)
        denominator = (
            (mu_x_sq + mu_y_sq + self.C1)
            * (sigma_x_sq + sigma_y_sq + self.C2)
        )

        # clamp_min is safer than simply adding a tiny epsilon in FP16 paths.
        denominator = denominator.clamp_min(self.eps)
        ssim_map = numerator / denominator
        _check_finite("SSIM map", ssim_map)

        # Numerical rounding can occasionally move SSIM slightly outside [-1, 1].
        ssim_value = ssim_map.mean().clamp(-1.0, 1.0)
        loss = 1.0 - ssim_value
        _check_finite("SSIM loss", loss)
        return loss


class GradientLoss(nn.Module):
    """AMP-safe Sobel gradient reconstruction loss."""

    def __init__(self, loss_type="l1"):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type not in {"l1", "mse"}:
            raise ValueError("loss_type must be 'l1' or 'mse'")

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _compute_gradients(self, x):
        channels = x.shape[1]
        kernel_x = self.sobel_x.to(device=x.device, dtype=x.dtype).repeat(
            channels, 1, 1, 1
        )
        kernel_y = self.sobel_y.to(device=x.device, dtype=x.dtype).repeat(
            channels, 1, 1, 1
        )
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
        return grad_x, grad_y

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} does not match "
                f"target shape {target.shape}"
            )

        _check_finite("prediction before GradientLoss", prediction)
        _check_finite("target before GradientLoss", target)

        # Compute Sobel response in FP32 under autocast as well.
        pred = prediction.float()
        tgt = target.float()

        pred_gx, pred_gy = self._compute_gradients(pred)
        target_gx, target_gy = self._compute_gradients(tgt)

        if self.loss_type == "l1":
            loss_x = F.l1_loss(pred_gx, target_gx)
            loss_y = F.l1_loss(pred_gy, target_gy)
        else:
            loss_x = F.mse_loss(pred_gx, target_gx)
            loss_y = F.mse_loss(pred_gy, target_gy)

        loss = 0.5 * (loss_x + loss_y)
        _check_finite("gradient loss", loss)
        return loss


class CompositeReconstructionLoss(nn.Module):
    """MSE + SSIM + gradient reconstruction objective with finite checks."""

    def __init__(
        self,
        lambda_mse=0.5,
        lambda_ssim=1.0,
        lambda_grad=0.1,
        ssim_window_size=11,
        ssim_sigma=1.5,
        data_range=1.0,
        gradient_loss_type="l1",
    ):
        super().__init__()
        self.lambda_mse = float(lambda_mse)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_grad = float(lambda_grad)

        self.ssim_loss = SSIMLoss(
            window_size=ssim_window_size,
            sigma=ssim_sigma,
            data_range=data_range,
        )
        self.gradient_loss = GradientLoss(loss_type=gradient_loss_type)

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} does not match "
                f"target shape {target.shape}"
            )

        _check_finite("prediction entering composite loss", prediction)
        _check_finite("target entering composite loss", target)

        # Critical AMP fix: MSE can overflow in half precision for large activations.
        pred32 = prediction.float()
        target32 = target.float()
        loss_mse = F.mse_loss(pred32, target32)
        _check_finite("MSE loss", loss_mse)

        loss_ssim = self.ssim_loss(prediction, target)
        loss_grad = self.gradient_loss(prediction, target)

        total_loss = (
            self.lambda_mse * loss_mse
            + self.lambda_ssim * loss_ssim
            + self.lambda_grad * loss_grad
        )
        _check_finite("total reconstruction loss", total_loss)

        return {
            "total_loss": total_loss,
            "mse_loss": loss_mse,
            "ssim_loss": loss_ssim,
            "gradient_loss": loss_grad,
        }
