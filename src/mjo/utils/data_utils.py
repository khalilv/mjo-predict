import torch
import numpy as np
from typing import Optional

def prep_input(in_data: torch.Tensor, in_date_encodings: torch.Tensor, out_date_encodings: torch.Tensor, forecast_data: Optional[torch.Tensor], out_timestamps: np.ndarray, forecast_timestamps: Optional[np.ndarray]):
    """Prepare input tensors for MJO forecasting model.

    Constructs three input components:
    - x_past: Historical observations with optional date encodings
    - x_future: Future forecast ensemble members (if available) with date encodings
    - x_static: MJO phase computed from latest RMM1/RMM2 values

    Args:
        in_data: Historical RMM data of shape (B, T, V)
        in_date_encodings: Historical date encodings of shape (B, T, D)
        out_date_encodings: Future date encodings of shape (B, T, D)
        forecast_data: Optional ensemble forecast data of shape (B, E, T, V)
        out_timestamps: Future timestamps
        forecast_timestamps: Optional forecast timestamps (must match out_timestamps)

    Returns:
        Tuple of (x_past, x_future, x_static) where:
        - x_past: Historical inputs of shape (B, T, V+D)
        - x_future: Future inputs of shape (B, T, E*V+D) or (B, T, D)
        - x_static: MJO phase of shape (B, 1, 1)
    """
    # Process forecast ensemble data if available
    if forecast_data is not None:
        assert (forecast_timestamps == out_timestamps).all(), 'Found mismatch between forecast timestamps and out timestamps'
        forecast_data = forecast_data.permute(0, 2, 1, 3) # (B, T, E, V)
        forecast_future = forecast_data.flatten(2, 3) # (B, T, E*V) flatten ensemble members to separate variables
        x_future = torch.cat([forecast_future, out_date_encodings], dim=-1) if out_date_encodings is not None else forecast_future
    else:
        x_future = out_date_encodings

    # Concatenate historical data with date encodings
    x_past =  torch.cat([in_data, in_date_encodings], dim=-1) if in_date_encodings is not None else in_data

    # Compute MJO phase (1-8) from latest RMM1/RMM2 using arctan2
    # Convert angle to degrees [0, 360), then bin into 8 phases of 45° each
    angle = (torch.atan2(in_data[:, -1, 1], in_data[:, -1, 0]) * 180 / torch.pi + 180) % 360
    phase = torch.floor(angle/45) + 1
    x_static = phase.unsqueeze(1).unsqueeze(1)

    return (x_past, x_future, x_static)