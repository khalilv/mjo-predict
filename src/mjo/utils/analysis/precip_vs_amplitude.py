import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from mjo.utils.RMM.io import load_rmm_indices
from matplotlib import pyplot as plt
import xarray as xr



start_date = '2014-06-01'
end_date = '2021-06-01'
zoom_start = '2016-12-01'
zoom_end = '2017-04-01'

ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"    

# California precip: full range
era5 = xr.open_zarr('/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr')
global_precip = era5['total_precipitation_24hr'].sel(time=slice(start_date, end_date))
california_precip = global_precip.sel(latitude=slice(32.534, 42.009), longitude=slice(235.59, 245.869))
california_precip_mean = california_precip.mean(dim=['latitude', 'longitude'])
precip_dates = california_precip_mean.time.values
precip_values = california_precip_mean.values * 1000

# MJO amplitude: zoomed in
ground_truth_ds_zoom = load_rmm_indices(ground_truth_path).loc[zoom_start:zoom_end]
amplitudes_zoom = ground_truth_ds_zoom.amplitude.values
dates_zoom = ground_truth_ds_zoom.index.values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top subplot: California Precipitation (full range)
color1 = 'tab:red'
ax1.plot(precip_dates, precip_values, color=color1, label='California Precip', alpha=0.7)
ax1.set_ylabel('24hr precipitation (mm)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_title(f'California 24hr Precipitation ({start_date} to {end_date})')
ax1.grid(True)

# Add outline for zoomed region in top plot
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Convert zoom_start and zoom_end to matplotlib date numbers
zoom_start_num = mdates.date2num(pd.to_datetime(zoom_start))
zoom_end_num = mdates.date2num(pd.to_datetime(zoom_end))

# Get y-limits for the rectangle
ymin, ymax = ax1.get_ylim()

# Add rectangle outline (no fill) for the zoomed region
rect = Rectangle(
    (zoom_start_num, ymin),
    width=zoom_end_num - zoom_start_num,
    height=ymax - ymin,
    linewidth=2,
    edgecolor='black',
    facecolor='none',
    linestyle='--',
    zorder=10,
)
ax1.add_patch(rect)

# Ensure x-axis is in date format
ax1.xaxis_date()

# Bottom subplot: MJO Amplitude (zoomed in)
color2 = 'tab:blue'
ax2.plot(dates_zoom, amplitudes_zoom, color=color2, label='MJO Amplitude')
# Add dashed black line at amplitude = 1
ax2.axhline(y=1, color='darkgrey', linestyle='--', linewidth=1.5, label='Amplitude = 1')
ax2.set_ylabel('Amplitude', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_xlabel('Date')
ax2.set_title(f'MJO Amplitude ({zoom_start} to {zoom_end})')
ax2.grid(True)

fig.tight_layout()
plt.savefig('amplitude_vs_precip_2017.png')
