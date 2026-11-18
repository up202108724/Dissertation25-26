from window_forecasting import fit_and_forecast, reseed

from .patchtst import PatchTST


def run_patchtst_baseline(*, seed, lookback, train_scaled, val_scaled, scaler, horizon,
                          device, epochs, patience, batch_size, lr, dropout=0.2,
                          use_scheduler=True):
    """Univariate PatchTST: train on the product's windows, then forecast
    ``horizon`` days recursively.  Returns ``(forecast, train_losses, val_losses)``
    with the forecast in raw target units."""
    reseed(seed)
    model = PatchTST(lookback=lookback, dropout=dropout)
    return fit_and_forecast(
        model, lookback=lookback, train_scaled=train_scaled, val_scaled=val_scaled,
        scaler=scaler, horizon=horizon, device=device, epochs=epochs,
        patience=patience, batch_size=batch_size, lr=lr, use_scheduler=use_scheduler,
    )
