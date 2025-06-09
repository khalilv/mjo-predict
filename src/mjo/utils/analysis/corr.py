import numpy as np
from tqdm import tqdm
from mjo.utils.plot import lag_correlation_plot

output_filename = '/Users/kvirji/Documents/mjo-predict/plots/lag_correlation.png'
dataset_path = '/Users/kvirji/Documents/mjo-predict/DATA/MJO/preprocessed/reference_period_1979-09-07_to_2001-12-31/train.npz'
statistics_path = '/Users/kvirji/Documents/mjo-predict/DATA/MJO/preprocessed/reference_period_1979-09-07_to_2001-12-31/statistics.npz'
variables = ["RMM1", "RMM2"] 
max_lag = 1440

data = dict(np.load(dataset_path))
data = np.array([data[v] for v in variables]).T

stats = dict(np.load(statistics_path))
means = np.array([stats[f"{v}_mean"] for v in variables])
stds = np.array([stats[f"{v}_std"] for v in variables])
data_norm = (data - means) / stds

data_norm = data_norm.T
V, T = data_norm.shape  
correlations = []
 
for step in tqdm(range(max_lag)):
    lags = data_norm[:, :-step] if step > 0 else data_norm
    targets = data_norm[:, step:]
    corr = []
    for i in range(V):
        x, y = lags[i], targets[i]
        valid = ~np.isnan(x) & ~np.isnan(y)
        v = np.corrcoef(x[valid], y[valid])[0, 1]
        corr.append(v)
    correlations.append(corr)
    
correlations = np.array(correlations)

lag_correlation_plot(
    corrs=correlations.T,
    timesteps=np.arange(max_lag),
    variables=['RMM1', 'RMM2'],
    title='MJO correlation',
    output_filename=output_filename
)