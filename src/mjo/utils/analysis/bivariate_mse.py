import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import bivariate_mse_vs_lead_time_plot


def compute_bmse(predict_rmm1, ground_truth_rmm1, predict_rmm2, ground_truth_rmm2):
    predict_amplitude = np.sqrt(np.square(predict_rmm1) + np.square(predict_rmm2))
    predict_phase = np.arctan2(predict_rmm2, predict_rmm1)
    ground_truth_amplitude = np.sqrt(np.square(ground_truth_rmm1) + np.square(ground_truth_rmm2))
    ground_truth_phase = np.arctan2(ground_truth_rmm2, ground_truth_rmm1)

    bmse = np.mean(np.square(predict_rmm1 - ground_truth_rmm1) + np.square(predict_rmm2 - ground_truth_rmm2))
    bmsea = np.mean(np.square(predict_amplitude - ground_truth_amplitude))
    bmsep = np.mean(2*predict_amplitude*ground_truth_amplitude*(1-np.cos(predict_phase - ground_truth_phase)))
    assert np.isclose(bmse, bmsea + bmsep), f'Found mismatch between BMSE {bmse} and components BMSEa {bmsea}, BMSEp {bmsep}'
    return bmsea, bmsep

def main():

    deterministic_predict_dirs = [
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/TSMixer/FuXi/no_hist/logs/version_0/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/mock_u250'
    ]
    ensemble_predict_dirs = [
        '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi',
    ]
    deterministic_labels = [
        # 'TSMixer no hist',
        # 'U250'
    ]
    ensemble_labels = [
        'FuXi'
    ]
    ensemble_members = ['mean']
    
    ground_truth_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
    output_filename = '/glade/derecho/scratch/kvirji/mjo-predict/plots/bmse_fuxi_u250_mean.png'

    ground_truth_df = load_rmm_indices(ground_truth_filepath)

    all_lead_time_amplitude_errors = []
    all_lead_time_phase_errors = []
    max_lead_times = []
    
    # deterministic forecasts
    for predict_dir in deterministic_predict_dirs:
        dataframes = []
        max_lead_time = -1
        for filename in tqdm(os.listdir(predict_dir), f'Loading data from {predict_dir}'):
            df = load_rmm_indices(os.path.join(predict_dir, filename))
            max_lead_time = max(max_lead_time, len(df))
            dataframes.append(df)

        lead_time_amplitude_errors = []
        lead_time_phase_errors = []
        for lt in tqdm(range(max_lead_time), 'Computing corr per lead time'):
            predict_group = []
            ground_truth_group = []

            for df in dataframes:
                if lt < len(df):
                    predict_row = df.iloc[lt]
                    ground_truth_row = ground_truth_df.loc[df.index[lt]]
                    predict_group.append(predict_row)
                    ground_truth_group.append(ground_truth_row)

            predict_group = pd.DataFrame(predict_group)
            ground_truth_group = pd.DataFrame(ground_truth_group)
            bmsea, bmsep = compute_bmse(predict_group.RMM1.values, ground_truth_group.RMM1.values, predict_group.RMM2.values, ground_truth_group.RMM2.values)
            lead_time_amplitude_errors.append(bmsea)
            lead_time_phase_errors.append(bmsep)
        
        all_lead_time_amplitude_errors.append(lead_time_amplitude_errors)
        all_lead_time_phase_errors.append(lead_time_phase_errors)
        max_lead_times.append(max_lead_time)

    # ensemble forecasts
    ensemble_member_labels = []
    for label_idx, predict_dir in enumerate(ensemble_predict_dirs):
        for member in ensemble_members:
            dataframes = []
            max_lead_time = -1
            for filename in tqdm(os.listdir(predict_dir), f'Loading {member} data from {predict_dir}'):
                member_file_path = os.path.join(predict_dir, filename, f'{member}.txt')
                if os.path.exists(member_file_path):
                    df = load_rmm_indices(member_file_path)
                    max_lead_time = max(max_lead_time, len(df))
                    dataframes.append(df)

            lead_time_amplitude_errors = []
            lead_time_phase_errors = []
            for lt in tqdm(range(max_lead_time), 'Computing corr per lead time'):
                predict_group = []
                ground_truth_group = []

                for df in dataframes:
                    if lt < len(df):
                        predict_row = df.iloc[lt]
                        ground_truth_row = ground_truth_df.loc[df.index[lt]]
                        predict_group.append(predict_row)
                        ground_truth_group.append(ground_truth_row)

                predict_group = pd.DataFrame(predict_group)
                ground_truth_group = pd.DataFrame(ground_truth_group)
                bmsea, bmsep = compute_bmse(predict_group.RMM1.values, ground_truth_group.RMM1.values, predict_group.RMM2.values, ground_truth_group.RMM2.values)
                lead_time_amplitude_errors.append(bmsea)
                lead_time_phase_errors.append(bmsep)
            
            all_lead_time_amplitude_errors.append(lead_time_amplitude_errors)
            all_lead_time_phase_errors.append(lead_time_phase_errors)
            max_lead_times.append(max_lead_time)
            ensemble_member_labels.append(f'{ensemble_labels[label_idx]} {member}')

    bivariate_mse_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        bmsea=all_lead_time_amplitude_errors,
        bmsep=all_lead_time_phase_errors,
        labels=deterministic_labels + ensemble_member_labels,
        output_filename=output_filename
    )

if __name__ == "__main__":
    main()
