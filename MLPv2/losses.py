"""
losses.py — All loss function definitions for the MLP forecaster.

Usage
-----
    from losses import get_loss
    loss_fn = get_loss('quantile_0.9')   # or 'mse', 'mae', 'huber', 'tweedie', etc.

Supported keys (pass as loss_type string)
-----------------------------------------
    mse              — Mean Squared Error
    mae              — Mean Absolute Error (L1)
    huber            — Huber loss  (delta=1.0)
    logcosh          — Log-Cosh (smooth MAE approximation)
    quantile_<q>     — Pinball / Quantile loss  e.g. 'quantile_0.5', 'quantile_0.9'
    smape            — Symmetric Mean Absolute Percentage Error
    tweedie_<p>      — Tweedie loss  e.g. 'tweedie_1.5'  (p in (1, 2))
    wmse_<w>         — Recency-weighted MSE  e.g. 'wmse_2.0'  (exponential weight)
    mase             — Mean Absolute Scaled Error (requires train_mean kwarg)
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# 1. MSE — Mean Squared Error
#    Optimises the conditional mean. Sensitive to large spikes.
# ---------------------------------------------------------------------------
class MSELoss(nn.MSELoss):
    """Standard PyTorch MSE. Included for completeness."""
    pass


# ---------------------------------------------------------------------------
# 2. MAE — Mean Absolute Error (L1)
#    Optimises the conditional median. Robust to spikes and intermittency.
# ---------------------------------------------------------------------------
class MAELoss(nn.L1Loss):
    """Standard PyTorch L1 / MAE. Included for completeness."""
    pass


# ---------------------------------------------------------------------------
# 3. Huber Loss
#    Quadratic for small errors, linear for large ones.
#    delta controls the transition point (default=1.0 in scaled space).
# ---------------------------------------------------------------------------
class HuberLoss(nn.HuberLoss):
    """Standard PyTorch Huber. Included for completeness."""
    def __init__(self, delta: float = 1.0):
        super().__init__(delta=delta)


# ---------------------------------------------------------------------------
# 4. Log-Cosh Loss
#    log(cosh(pred - target))
#    Behaves like MSE for small errors, like MAE for large ones.
#    Fully differentiable everywhere (unlike Huber's kink).
# ---------------------------------------------------------------------------
class LogCoshLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(diff + torch.nn.functional.softplus(-2.0 * diff) - torch.log(torch.tensor(2.0)))


# ---------------------------------------------------------------------------
# 5. Quantile / Pinball Loss
#    q=0.5  → identical to MAE (median forecast)
#    q>0.5  → penalises under-forecasting more  (e.g. q=0.9 for safety stock)
#    q<0.5  → penalises over-forecasting more
#    Training at multiple quantiles gives a distribution-free prediction interval.
# ---------------------------------------------------------------------------
class QuantileLoss(nn.Module):
    def __init__(self, q: float = 0.5):
        super().__init__()
        if not 0.0 < q < 1.0:
            raise ValueError(f"Quantile q must be in (0, 1), got {q}.")
        self.q = q

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        e = target - pred
        return torch.mean(torch.max(self.q * e, (self.q - 1.0) * e))

    def __repr__(self):
        return f"QuantileLoss(q={self.q})"


# ---------------------------------------------------------------------------
# 6. sMAPE — Symmetric Mean Absolute Percentage Error
#    Scale-free: useful when comparing across products with very different volumes.
#    Range [0, 2]. Numerically stable (avoids division by zero via epsilon).
#    NOTE: undefined semantics when both pred and target are 0 → use with care
#    for intermittent series. Prefer Tweedie or MAE in that case.
# ---------------------------------------------------------------------------
class SMAPELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denom = (torch.abs(pred) + torch.abs(target)) / 2.0 + self.eps
        return torch.mean(torch.abs(pred - target) / denom)


# ---------------------------------------------------------------------------
# 7. Tweedie Loss
#    p in (1, 2):  p→1 behaves like Poisson (good for counts / sparse series)
#                  p→2 behaves like Gamma  (good for continuous positive series)
#    Handles zeros natively. Asymmetric: penalises under-forecasting more.
#    Best used on UNSCALED (or log-transformed) non-negative targets.
#    Used as the default objective for retail demand in the M5 competition.
# ---------------------------------------------------------------------------
class TweedieLoss(nn.Module):
    def __init__(self, p: float = 1.5):
        super().__init__()
        if not 1.0 < p < 2.0:
            raise ValueError(f"Tweedie power p must be in (1, 2), got {p}.")
        self.p = p

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, min=1e-6)   # predictions must be positive
        a = target * pred.pow(1.0 - self.p) / (1.0 - self.p)
        b = pred.pow(2.0 - self.p) / (2.0 - self.p)
        return torch.mean(-a + b)

    def __repr__(self):
        return f"TweedieLoss(p={self.p})"


# ---------------------------------------------------------------------------
# 8. Recency-Weighted MSE
#    Applies exponentially increasing weights over the sequence so that
#    errors near the end of the forecast horizon are penalised more.
#    weight_t = exp(alpha * t / T),  alpha controls how steeply weights grow.
#    Useful when later forecast errors are more costly (e.g. production planning).
# ---------------------------------------------------------------------------
class RecencyWeightedMSELoss(nn.Module):
    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        N = pred.shape[0]
        t = torch.arange(N, dtype=pred.dtype, device=pred.device)
        weights = torch.exp(self.alpha * t / float(N))
        weights = weights / weights.sum()
        return torch.sum(weights * (pred.flatten() - target.flatten()) ** 2)

    def __repr__(self):
        return f"RecencyWeightedMSELoss(alpha={self.alpha})"


# ---------------------------------------------------------------------------
# 9. MASE — Mean Absolute Scaled Error
#    Scales MAE by the in-sample naive (lag-1) MAE, making it comparable
#    across products with different volumes.
#    train_mean must be provided at construction time (scalar float).
#    MASE < 1 means you beat the naive forecast.
# ---------------------------------------------------------------------------
class MASELoss(nn.Module):
    def __init__(self, naive_mae: float):
        super().__init__()
        if naive_mae <= 0:
            raise ValueError("naive_mae (in-sample lag-1 MAE) must be > 0.")
        self.naive_mae = naive_mae

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target)) / self.naive_mae

    def __repr__(self):
        return f"MASELoss(naive_mae={self.naive_mae:.4f})"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def get_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Return the loss module corresponding to `loss_type`.

    Examples
    --------
    get_loss('mse')
    get_loss('huber', delta=0.5)
    get_loss('quantile_0.9')
    get_loss('tweedie_1.5')
    get_loss('wmse_2.0')
    get_loss('mase', naive_mae=3.7)
    """
    key = loss_type.lower().strip()

    if key == 'mse':
        return MSELoss()
    elif key == 'mae':
        return MAELoss()
    elif key in ('huber', 'smoothl1'):
        return HuberLoss(delta=kwargs.get('delta', 1.0))
    elif key == 'logcosh':
        return LogCoshLoss()
    elif key == 'smape':
        return SMAPELoss()
    elif key.startswith('quantile_'):
        q = float(key.split('_', 1)[1])
        return QuantileLoss(q=q)
    elif key.startswith('tweedie_'):
        p = float(key.split('_', 1)[1])
        return TweedieLoss(p=p)
    elif key.startswith('wmse_'):
        alpha = float(key.split('_', 1)[1])
        return RecencyWeightedMSELoss(alpha=alpha)
    elif key == 'mase':
        naive_mae = kwargs.get('naive_mae')
        if naive_mae is None:
            raise ValueError("get_loss('mase') requires naive_mae=<float> kwarg.")
        return MASELoss(naive_mae=float(naive_mae))
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            "Supported: mse, mae, huber, logcosh, smape, "
            "quantile_<q>, tweedie_<p>, wmse_<alpha>, mase."
        )
