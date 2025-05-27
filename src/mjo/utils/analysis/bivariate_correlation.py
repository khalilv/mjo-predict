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

    predict_dir = '/glade/derecho/scratch/kvirji/mjo-predict/exps/mock'
    ground_truth_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ABM/rmm.74toRealtime.txt"
    output_filename = '/glade/derecho/scratch/kvirji/mjo-predict/plots/bivariate_correlation_ours_vs_abm.png'

    ground_truth_df = load_rmm_indices(ground_truth_filepath)

    dataframes = []
    max_lead_time = -1
    for filename in tqdm(os.listdir(predict_dir), 'Loading data'):
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

    bivariate_correlation_vs_lead_time_plot(
        lead_times=np.arange(1, max_lead_time + 1),
        correlations=[lead_time_correlations],
        labels=['Ours'],
        output_filename=output_filename
    )

if __name__ == "__main__":
    main()
