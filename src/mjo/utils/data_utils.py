import torch
import numpy as np
from typing import Optional, Any

def prep_input(in_data: torch.Tensor, forecast_data: Optional[torch.Tensor], in_timestamps: np.ndarray, out_timestamps: np.ndarray, forecast_timestamps: Optional[np.ndarray], device: torch.device, dtype: torch.dtype, year_normalization: Any = None):
    
    in_timestamp_encodings = get_timestamp_encodings(in_timestamps, device, dtype, year_normalization)
    out_timestamp_encodings = get_timestamp_encodings(out_timestamps, device, dtype, year_normalization)

    if forecast_data is not None:
        assert (forecast_timestamps == out_timestamps[:, :forecast_timestamps.shape[1]]).all(), 'Found mismatch between forecast timestamps and out timestamps'
        forecast_data = forecast_data.permute(0, 2, 1, 3) # (B, T, E, V)
        forecast_future = forecast_data.flatten(2, 3) # (B, T, E*V) flatten ensemble members to seperate variables
        if forecast_future.shape[1] < out_timestamps.shape[1]:
            padding = torch.zeros((forecast_future.shape[0], out_timestamps.shape[1] - forecast_future.shape[1], forecast_future.shape[2]), device=device, dtype=dtype)
            forecast_future = torch.cat([forecast_future, padding], dim=1)
        x_past =  torch.cat([in_data, in_timestamp_encodings], dim=-1)
        x_future = torch.cat([out_timestamp_encodings, forecast_future], dim=-1) 
    else:
        x_past =  torch.cat([in_data, in_timestamp_encodings], dim=-1) 
        x_future = out_timestamp_encodings
    
    x_static = None    
    return (x_past, x_future, x_static)

def get_timestamp_encodings(timestamps, device, dtype, year_normalization=None):
    years = torch.tensor(timestamps.astype('datetime64[Y]').astype(int) + 1970, device=device, dtype=dtype)
    if year_normalization is not None:
        years = year_normalization.normalize(years)

    start_of_year = timestamps.astype('datetime64[Y]')
    doy = (timestamps - start_of_year).astype('timedelta64[D]').astype(int)
    next_year = (start_of_year + np.timedelta64(1, 'Y')).astype('datetime64[D]')
    days_in_year = (next_year - start_of_year).astype(int)  # shape (B, L)

    # Angle for sin/cos embedding
    angle = 2 * np.pi * doy / days_in_year
    sin = np.sin(angle)
    cos = np.cos(angle)

    doy_encoding = torch.tensor(np.stack([sin, cos], axis=-1),  device=device, dtype=dtype)

    timestamp_encodings = torch.cat([years.unsqueeze(-1), doy_encoding], dim=-1)

    return timestamp_encodings