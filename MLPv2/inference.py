import torch
import numpy as np
from typing import Optional

def recursive_inference(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,  
    future_exog: np.ndarray,     
    target_channel: int = 0,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Recursively forecasts `horizon` steps using a 1-step-ahead model.
    Uses known `future_exog` for the next step's input.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    horizon = len(future_exog)
    
    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    # Identificar índices exógenos de forma segura
    C_in = recent_history.shape[1]
    exog_indices = [idx for idx in range(C_in) if idx != target_channel]

    # Escalar o histórico recente (apenas o target, assumindo exogs já escaladas)
    current_x_scaled = recent_history.copy()
    current_x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )
    
    preds_scaled = []
    model = model.to(device).eval()

    # Alinhar janela inicial com a mesma lógica do make_windows
    # X[t] = target[t], exog[t+1]
    input_window = current_x_scaled.copy()
    if len(exog_indices) > 0:
        # Fazer shift das exógenas
        input_window[:-1, exog_indices] = current_x_scaled[1:, exog_indices]
        # Inserir exógena do dia que vamos prever no último step
        input_window[-1, exog_indices] = future_exog[0]
    
    with torch.no_grad(): # BOA PRÁTICA: Desligar gradientes na inferência
        for i in range(horizon):

            x_tensor = torch.from_numpy(input_window).float().unsqueeze(0).to(device) # (1, L, C_in)
            
            y_pred = model(x_tensor)
            val_pred = y_pred.item()
            preds_scaled.append(val_pred)

            # --- Shift da Janela ---
            input_window = np.roll(input_window, -1, axis=0)
            
            # 1. O novo target no último step é a previsão que acabámos de fazer
            input_window[-1, target_channel] = val_pred
            
            # 2. Inserir as exógenas do PRÓXIMO dia no último step
            if len(exog_indices) > 0 and (i + 1) < horizon:
                input_window[-1, exog_indices] = future_exog[i + 1]
            
    # Reshape para fazer inverse_transform
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds