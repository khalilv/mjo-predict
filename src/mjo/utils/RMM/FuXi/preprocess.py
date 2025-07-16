import os
import numpy as np
from tqdm import tqdm
from mjo.utils.RMM.io import load_rmm_indices

def main():
        
    forecast_dir = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi"
    output_dir = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/preprocessed/FuXi"

    os.makedirs(output_dir, exist_ok=True)

    for start_date in sorted(os.listdir(forecast_dir)):
        root = os.path.join(forecast_dir, start_date)
        RMM1, RMM2, phase, amplitude, phase_sin, phase_cos, doy_sin, doy_cos, dates = [], [], [], [], [], [], [], [], []
        RMM1_mean, RMM2_mean, phase_mean, amplitude_mean, phase_sin_mean, phase_cos_mean, doy_sin_mean, doy_cos_mean, dates_mean = [], [], [], [], [], [], [], [], []
        members = sorted(os.listdir(root))
        if len(members) < 52: print(f'Warning: only found {len(members)} members for {start_date}')
        for member in tqdm(members, f'Processing members for {start_date}'):
            member_df = load_rmm_indices(os.path.join(root, member))
            p = np.arctan2(member_df['RMM2'],member_df['RMM1'])
            member_df['phase_sin'] = np.sin(p)
            member_df['phase_cos'] = np.cos(p)
    
            doy = member_df.index.day_of_year
            angle = 2 * np.pi * doy / 366
            member_df['doy_sin'] = np.sin(angle)
            member_df['doy_cos'] = np.cos(angle)

            if member == 'mean.txt':
                RMM1_mean.append(member_df['RMM1'].values)
                RMM2_mean.append(member_df['RMM2'].values)
                amplitude_mean.append(member_df['amplitude'].values)
                phase_mean.append(member_df['phase'].values)
                phase_sin_mean.append(member_df['phase_sin'].values)
                phase_cos_mean.append(member_df['phase_cos'].values)
                doy_sin_mean.append(member_df['doy_sin'].values)
                doy_cos_mean.append(member_df['doy_cos'].values)
                dates_mean.append(member_df.index.values)
            else:
                RMM1.append(member_df['RMM1'].values)
                RMM2.append(member_df['RMM2'].values)
                amplitude.append(member_df['amplitude'].values)
                phase.append(member_df['phase'].values)
                phase_sin.append(member_df['phase_sin'].values)
                phase_cos.append(member_df['phase_cos'].values)
                doy_sin.append(member_df['doy_sin'].values)
                doy_cos.append(member_df['doy_cos'].values)
                dates.append(member_df.index.values)
        
        assert np.all(dates == dates[0]), f'Found non-matching dates within ensemble members in {root}'
        np.savez(os.path.join(output_dir, f'{start_date}_members.npz'),
            RMM1=np.array(RMM1),
            RMM2=np.array(RMM2),
            phase=np.array(phase),
            amplitude=np.array(amplitude),
            phase_sin=np.array(phase_sin),
            phase_cos=np.array(phase_cos),
            doy_sin=np.array(doy_sin),
            doy_cos=np.array(doy_cos),
            dates=np.array(dates[0]))

        assert np.all(dates[0] == dates_mean[0]), f'Found non-matching dates within ensemble mean in {root}'
        np.savez(os.path.join(output_dir, f'{start_date}_mean.npz'),
            RMM1=np.array(RMM1_mean),
            RMM2=np.array(RMM2_mean),
            phase=np.array(phase_mean),
            amplitude=np.array(amplitude_mean),
            phase_sin=np.array(phase_sin_mean),
            phase_cos=np.array(phase_cos_mean),
            doy_sin=np.array(doy_sin_mean),
            doy_cos=np.array(doy_cos_mean),
            dates=np.array(dates_mean[0]))

if __name__ == "__main__":
    main()
