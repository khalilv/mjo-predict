import os
from datetime import datetime, timedelta
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import phase_space_plot

def main():
        
    predict_dirs = [
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/TSMixer/FuXi_test2017-2021/ensemble_mean/no_hist/logs/version_0/outputs'
    ]
    labels = ['TSMixer']
    ensemble_dirs = [
        '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi/'
    ]
    ensemble_labels = ['FuXi']
    ensemble_members = ['mean'] #+ [f"{i:02d}" for i in range(0, 51, 5)]

    abm_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ABM/rmm.74toRealtime.txt"
    gt_filepath = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt'

    output_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/plots/phase_space'
    start_date = "2019-04-19"
    trajectory_length = 42

    os.makedirs(output_dir, exist_ok=True)

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = start_date_dt + timedelta(days=trajectory_length - 1)

    gt_df = load_rmm_indices(gt_filepath)
    abm_df = load_rmm_indices(abm_filepath)
    gt_slice = gt_df.loc[start_date_dt:end_date_dt]
    abm_slice = abm_df.loc[start_date_dt:end_date_dt]

    assert not gt_slice.isnull().values.any(), "NaNs found in ground truth dataframe slice"
    assert not abm_slice.isnull().values.any(), "NaNs found in abm dataframe slice"
    assert (gt_slice.index == abm_slice.index).all(), "Ground truth dataframe slice and abm dataframe slice do not contain the same dates"

    predict_dfs = []
    init_date = str(start_date_dt - timedelta(days=1)).split(' ')[0]
    for dir in predict_dirs:
        pred_df = load_rmm_indices(os.path.join(dir, f'{init_date}.txt'))
        pred_df_slice = pred_df.loc[start_date:end_date_dt]
        assert not pred_df_slice.isnull().values.any(), f"NaNs found in predict dataframe slice in {dir}"
        assert (gt_slice.index == pred_df_slice.index).all(), f"Predict dataframe slice from {dir} and ground truth dataframe slice do not contain the same dates"
        predict_dfs.append(pred_df_slice)
    
    member_labels = []
    for label_idx, dir in enumerate(ensemble_dirs):
        for member in ensemble_members:
            pred_df = load_rmm_indices(os.path.join(dir, init_date, f'{member}.txt'))
            pred_df_slice = pred_df.loc[start_date:end_date_dt]
            assert not pred_df_slice.isnull().values.any(), f"NaNs found in predict dataframe slice in {dir}"
            assert (gt_slice.index == pred_df_slice.index).all(), f"Predict dataframe slice from {dir} and ground truth dataframe slice do not contain the same dates"
            predict_dfs.append(pred_df_slice)
            member_labels.append(f'{ensemble_labels[label_idx]} {member}')

    output_filename = f'{start_date}.png'
    phase_space_plot(
        pred_rmm1s=[p.RMM1.values for p in predict_dfs],
        gt_rmm1=gt_slice.RMM1.values,
        pred_rmm2s=[p.RMM2.values for p in predict_dfs],
        gt_rmm2=gt_slice.RMM2.values,
        gt_label='Ours',
        labels=labels + member_labels,
        abm_rmm1=abm_slice.RMM1.values,
        abm_rmm2=abm_slice.RMM2.values,
        title=f'{trajectory_length}-day trajectory phase space diagram',
        output_filename=os.path.join(output_dir, output_filename)
    )

if __name__ == "__main__":
    main()
