import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import bivariate_correlation_vs_lead_time_plot


def bivariate_correlation(predict_rmm1, ground_truth_rmm1, predict_rmm2, ground_truth_rmm2):
    n = np.sum((predict_rmm1 * ground_truth_rmm1) + (predict_rmm2 * ground_truth_rmm2))
    d1 = np.sqrt(np.sum(np.square(predict_rmm1) + np.square(predict_rmm2)))
    d2 = np.sqrt(np.sum(np.square(ground_truth_rmm1) + np.square(ground_truth_rmm2)))
    return n / (d1*d2)

def main():

    deterministic_predict_dirs = [
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/TFT/history_plus_dates/hist_10d_predict_60d/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/TFT/history_plus_dates/hist_720d_predict_60d/logs/version_0/outputs'
    ]
    ensemble_predict_dirs = [
        '/glade/derecho/scratch/kvirji/DATA/MJO/RMM/FuXi/reference_period_1979-09-07_to_2001-12-31',
    ]
    deterministic_labels = [
        'TFT 10d',
        'TFT 720d'
    ]
    ensemble_labels = [
        'FuXi'
    ]
    ensemble_members = np.arange(51)
    
    ground_truth_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/RMM/reference_period_1979-09-07_to_2001-12-31/rmm.txt"
    output_filename = '/glade/derecho/scratch/kvirji/mjo-predict/plots/bivariate_correlation_tft.png'

    ground_truth_df = load_rmm_indices(ground_truth_filepath)

    all_lead_time_correlations = []
    max_lead_times = []
    
    # deterministic forecasts
    for predict_dir in deterministic_predict_dirs:
        dataframes = []
        max_lead_time = -1
        for filename in tqdm(os.listdir(predict_dir), f'Loading data from {predict_dir}'):
            df = load_rmm_indices(os.path.join(predict_dir, filename))
            max_lead_time = max(max_lead_time, len(df))
            dataframes.append(df)

        lead_time_correlations = []
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
            corr = bivariate_correlation(predict_group.RMM1.values, ground_truth_group.RMM1.values, predict_group.RMM2.values, ground_truth_group.RMM2.values)
            lead_time_correlations.append(corr)
        
        all_lead_time_correlations.append(lead_time_correlations)
        max_lead_times.append(max_lead_time)

    # ensemble forecasts
    ensemble_member_labels = []
    for label_idx, predict_dir in enumerate(ensemble_predict_dirs):
        for member in ensemble_members:
            dataframes = []
            max_lead_time = -1
            member_str = f"{member:02d}"
            for filename in tqdm(os.listdir(predict_dir), f'Loading {member_str} data from {predict_dir}'):
                member_file_path = os.path.join(predict_dir, filename, f'{member_str}.txt')
                if os.path.exists(member_file_path):
                    df = load_rmm_indices(member_file_path)
                    max_lead_time = max(max_lead_time, len(df))
                    dataframes.append(df)

            lead_time_correlations = []
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
                corr = bivariate_correlation(predict_group.RMM1.values, ground_truth_group.RMM1.values, predict_group.RMM2.values, ground_truth_group.RMM2.values)
                lead_time_correlations.append(corr)
            
            all_lead_time_correlations.append(lead_time_correlations)
            max_lead_times.append(max_lead_time)
            ensemble_member_labels.append(f'{ensemble_labels[label_idx]} {member_str}')

    bivariate_correlation_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        correlations=all_lead_time_correlations,
        labels=deterministic_labels + ensemble_member_labels,
        output_filename=output_filename
    )

if __name__ == "__main__":
    main()
