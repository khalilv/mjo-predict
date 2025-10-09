import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from mjo.utils.RMM.io import load_rmm_indices
from matplotlib import pyplot as plt
import xarray as xr



start_date = '2016-12-01'
end_date = '2017-04-01'

ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"    

# California precip: full range
era5 = xr.open_zarr('/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr')
global_precip = era5['total_precipitation_24hr'].sel(time=slice(start_date, end_date))
california_precip = global_precip.sel(latitude=slice(32.534, 42.009), longitude=slice(235.59, 245.869))
california_precip_mean = california_precip.mean(dim=['latitude', 'longitude'])
precip_dates = pd.to_datetime(california_precip_mean.time.values)
precip_values = california_precip_mean.values * 1000  # mm

# MJO amplitude: full range
ground_truth_ds = load_rmm_indices(ground_truth_path).loc[start_date:end_date]
amplitudes = ground_truth_ds.amplitude.values
amplitude_dates = pd.to_datetime(ground_truth_ds.index.values)

fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot MJO amplitude (left y-axis)
color_amp = 'tab:blue'
ax1.plot(amplitude_dates, amplitudes, color=color_amp, label='MJO Amplitude', linewidth=2)
ax1.set_ylabel('MJO Amplitude', color=color_amp)
ax1.tick_params(axis='y', labelcolor=color_amp)
ax1.axhline(y=1, color='darkgrey', linestyle='--', linewidth=1.5)
ax1.set_xlabel('Date')
ax1.set_title(f'MJO Amplitude and California 24hr Precipitation ({start_date} to {end_date})')
ax1.grid(True, which='both', axis='both', alpha=0.3)

# Plot California precip as bars (right y-axis)
ax2 = ax1.twinx()
color_precip = 'tab:red'
ax2.bar(precip_dates, precip_values, width=1.0, color=color_precip, alpha=0.25, label='California precipitation')
ax2.set_ylabel('24hr precipitation (mm)', color=color_precip)
ax2.tick_params(axis='y', labelcolor=color_precip)

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

fig.tight_layout()
plt.savefig('amplitude_vs_precip_fullrange.png')
