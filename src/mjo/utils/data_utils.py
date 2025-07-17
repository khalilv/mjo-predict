import torch
import numpy as np
from typing import Optional

def prep_input(in_data: torch.Tensor, in_date_encodings: torch.Tensor, out_date_encodings: torch.Tensor, forecast_data: Optional[torch.Tensor], out_timestamps: np.ndarray, forecast_timestamps: Optional[np.ndarray]):
       
    if forecast_data is not None:
        assert (forecast_timestamps == out_timestamps).all(), 'Found mismatch between forecast timestamps and out timestamps'
        forecast_data = forecast_data.permute(0, 2, 1, 3) # (B, T, E, V)
        forecast_future = forecast_data.flatten(2, 3) # (B, T, E*V) flatten ensemble members to seperate variables
        x_future = torch.cat([forecast_future, out_date_encodings], dim=-1)
    else:
        x_future = out_date_encodings
    
    x_past =  torch.cat([in_data, in_date_encodings], dim=-1)
    x_static = None  
    return (x_past, x_future, x_static)