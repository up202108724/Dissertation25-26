from window_forecasting import fit_and_forecast, reseed

from .dlinear import DLinear


def run_dlinear_baseline(*, seed, lookback, train_scaled, val_scaled, scaler, horizon,
                         device, epochs, patience, batch_size, lr, kernel_size=7,
                         use_scheduler=True):
    """Univariate DLinear: train on the product's windows, then forecast
    ``horizon`` days recursively.  Returns ``(forecast, train_losses, val_losses)``
    with the forecast in raw target units."""
    reseed(seed)
    model = DLinear(lookback=lookback, kernel_size=kernel_size)
    return fit_and_forecast(
        model, lookback=lookback, train_scaled=train_scaled, val_scaled=val_scaled,
        scaler=scaler, horizon=horizon, device=device, epochs=epochs,
        patience=patience, batch_size=batch_size, lr=lr, use_scheduler=use_scheduler,
    )
