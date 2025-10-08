import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from mjo.utils.analysis.utils import load_forecast
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import (
    scatter_amplitudes_by_init_date_tripanel
)


start_date = '2017-01-01'
end_date = '2017-03-01'
ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
fuxi_path = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi'
output_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/plots/production/2019-2021/case_study/2017/'
month = 'JAN-FEB'
data = []
os.makedirs(output_dir, exist_ok=True)
ground_truth = load_rmm_indices(ground_truth_path)
forecasted_dfs, max_lt, _ = load_forecast(fuxi_path, start_date, end_date, 'mean')

for df in forecasted_dfs:
    gt = ground_truth.loc[df.index].amplitude.values
    fc = df.amplitude.values
    score = ((gt > 1).astype(int) & (fc > 1).astype(int)) | ((gt < 1).astype(int) & (fc < 1).astype(int))
    data.append(np.stack([fc, score], axis=1))
data = np.array(data)
dates = [df.index[0] - timedelta(days=1) for df in forecasted_dfs]
scatter_amplitudes_by_init_date_tripanel(data, dates, output_filename=os.path.join(output_dir, f'forecasted_amplitudes_fuxi_{month}'))
