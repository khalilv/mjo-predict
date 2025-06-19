import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import correlation_scatter_plot

def main():
        
    forecast_dir = "/glade/derecho/scratch/kvirji/DATA/MJO/RMM/FuXi/reference_period_1979-09-07_to_2001-12-31"
    output_dir = "/glade/derecho/scratch/kvirji/DATA/preprocessed/MJO/FuXi/reference_period_1979-09-07_to_2001-12-31"

    os.makedirs(output_dir, exist_ok=True)

    for start_date in sorted(os.listdir(forecast_dir)):
        root = os.path.join(forecast_dir, start_date)
        RMM1, RMM2, phase, amplitude, dates = [], [], [], [], []
        members = sorted(os.listdir(root))
        if len(members) < 51: print(f'Warning: only found {len(members)} members for {start_date}')
        for member in tqdm(members, f'Processing members for {start_date}'):
            member_df = load_rmm_indices(os.path.join(root, member))
            RMM1.append(member_df['RMM1'].values)
            RMM2.append(member_df['RMM2'].values)
            amplitude.append(member_df['amplitude'].values)
            phase.append(member_df['phase'].values)
            dates.append(member_df.index.values)
        
        assert np.all(dates == dates[0]), f'Found non-matching dates within ensemble members in {root}'
        np.savez(os.path.join(output_dir, f'{start_date}.npz'),
            RMM1=np.array(RMM1),
            RMM2=np.array(RMM2),
            phase=np.array(phase),
            amplitude=np.array(amplitude),
            dates=np.array(dates[0]))


if __name__ == "__main__":
    main()
