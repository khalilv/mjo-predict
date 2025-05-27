import os
from datetime import datetime, timedelta
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import phase_space_plot

def main():
        
    predict_filepath = '/glade/derecho/scratch/kvirji/DATA/MJO/RMM/reference_period_1979-09-07_to_2001-12-31/rmm.txt'
    ground_truth_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ABM/rmm.74toRealtime.txt"
    output_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/plots/phase_space'
    start_date = "2019-04-18"
    trajectory_length = 42

    os.makedirs(output_dir, exist_ok=True)

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = start_date_dt + timedelta(days=trajectory_length - 1)

    predict_df = load_rmm_indices(predict_filepath)
    ground_truth_df = load_rmm_indices(ground_truth_filepath)

    predict_slice = predict_df.loc[start_date_dt:end_date_dt]
    ground_truth_slice = ground_truth_df.loc[start_date_dt:end_date_dt]

    assert not predict_slice.isnull().values.any(), "NaNs found in predict dataframe slice"
    assert not ground_truth_slice.isnull().values.any(), "NaNs found in ground truth dataframe slice"
    assert (predict_slice.index == ground_truth_slice.index).all(), "Predict dataframe slice and ground_truth dataframe slice do not contain the same dates"

    output_filename = f'{start_date}.png'
    phase_space_plot(
        pred_rmm1s=[predict_slice.RMM1.values],
        gt_rmm1=ground_truth_slice.RMM1.values,
        pred_rmm2s=[predict_slice.RMM2.values],
        gt_rmm2=ground_truth_slice.RMM2.values,
        labels=['Ours'],
        title=f'{trajectory_length}-day trajectory phase space diagram',
        output_filename=os.path.join(output_dir, output_filename)
    )

if __name__ == "__main__":
    main()
