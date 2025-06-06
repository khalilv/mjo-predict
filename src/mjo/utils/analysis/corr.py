import glob
import os 
import numpy as np
from tqdm import tqdm
from mjo.utils.plot import lag_correlation_plot

output_path = '/home/khalilv/Documents/mjo-predict/plots/correlation_denorm_abm.png'
dataset_path = '/home/khalilv/Documents/mjo-predict/DATA/MJO/preprocessed/ABM/train.npz'
statistics_path = '/home/khalilv/Documents/mjo-predict/DATA/MJO/preprocessed/reference_period_1979-09-07_to_2001-12-31/statistics.npz'
variables = ["RMM1", "RMM2"] 
max_lag = 1440

data = dict(np.load(dataset_path))
data = np.array([data[v] for v in variables]).T

stats = dict(np.load(statistics_path))
means = np.array([stats[f"{v}_mean"] for v in variables])
stds = np.array([stats[f"{v}_std"] for v in variables])
data_norm = (data - means) / stds

means = []
stds = []
timesteps = []
for step in tqdm(range(max_lag)):
    lags = data_norm[:-step] if step > 0 else data_norm
    targets = data_norm[step:]
    vals = []
    for t in range(lags.shape[0]):
        if np.isnan(targets[t]).any() or np.isnan(lags[t]).any():
            continue
        v = np.corrcoef(targets[t], lags[t])[0, 1]
        vals.append(v)
    vals = np.array(vals)
    means.append(vals.mean())
    stds.append(vals.std())
    timesteps.append(step)

lag_correlation_plot(
    means=means,
    stds=stds,
    timesteps=timesteps,
    freq=1,
    title='MJO correlation',
    output_filename=output_filename
)