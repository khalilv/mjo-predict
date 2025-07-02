import numpy as np
import os
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import phase_space_composite_plot
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm

def compute_composite_by_phase(predict_dir, gt_df, member=None, trajectory_length=42, window=3, start_date=None):
    pred_phase_groups = defaultdict(list)
    gt_phase_groups = defaultdict(list)
    if start_date is not None:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    for fname in tqdm(sorted(os.listdir(predict_dir)), f'Loading composites for {predict_dir}'):
        if member is not None:
            init_date = datetime.strptime(fname, "%Y-%m-%d")
            init_date_str = str(init_date).split(' ')[0]
            file = os.path.join(predict_dir, init_date_str, f'{member}.txt')
        else:
            init_date = datetime.strptime(fname, "%Y-%m-%d.txt")
            init_date_str = str(init_date).split(' ')[0]
            file = os.path.join(predict_dir, f'{init_date_str}.txt')
       
        if start_date is not None and init_date < start_date:
            continue

        day_1 = init_date + timedelta(days=1)
        last_day = init_date + timedelta(days=trajectory_length)
        phase = int(gt_df.loc[day_1].phase)
        gt_slice = gt_df.loc[day_1:last_day]

        pred_df = load_rmm_indices(file)
        if window > 1: #smooth predictions and ground truth
            pred_df_smoothed = pred_df.rolling(window=window, center=False).mean()
            pred_df_smoothed[:window - 1] = pred_df[:window - 1]
            pred_df = pred_df_smoothed

            gt_slice_smoothed = gt_slice.rolling(window=window, center=False).mean()
            gt_slice_smoothed[:window - 1] = gt_slice[:window - 1]
            gt_slice = gt_slice_smoothed

        if len(pred_df) < trajectory_length:
            raise ValueError(f"Prediction dataframe in file: {file} has length: {len(pred_df)}, which is less than the required trajectory length: {trajectory_length}.")
        pred_df = pred_df.iloc[:trajectory_length] #clip to trajectory length
        
        assert not gt_slice.isnull().values.any(), "NaNs found in ground truth dataframe slice"
        assert not pred_df.isnull().values.any(), f"NaNs found in abm predict file: {file}"
        assert (gt_slice.index == pred_df.index).all(), "Ground truth dataframe slice and predict dataframe slice do not contain the same dates"

        pred_phase_groups[phase].append(pred_df)
        gt_phase_groups[phase].append(gt_slice)

    # composite trajectories
    pred_composite_trajectories = {}
    gt_composite_trajectories = {}
    for phase, dfs in pred_phase_groups.items():

        pred_rmm1 = np.stack([df.RMM1.values for df in dfs])
        pred_rmm2 = np.stack([df.RMM2.values for df in dfs])
        gt_rmm1 = np.stack([df.RMM1.values for df in gt_phase_groups[phase]])
        gt_rmm2 = np.stack([df.RMM2.values for df in gt_phase_groups[phase]])
        pred_composite_trajectories[phase] = (pred_rmm1.mean(axis=0), pred_rmm2.mean(axis=0))
        gt_composite_trajectories[phase] = (gt_rmm1.mean(axis=0), gt_rmm2.mean(axis=0))

    return pred_composite_trajectories, gt_composite_trajectories

def main():
    
    smoothing_window = 3
    start_date = "2020-01-01"
    predict_dirs = [
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/TFT/FuXi/ensemble_mean/no_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/TSMixer/FuXi/ensemble_mean/no_hist/logs/version_0/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/TSMixer/FuXi/ensemble_mean/MAE/no_hist/logs/version_0/outputs',
    ]
    labels = [#'TFT', 
              'TSMixer', 
              #'TSMixer MAE'
              ]
    ensemble_dirs = [
        '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi/'
    ]
    ensemble_labels = ['FuXi']
    ensemble_members = ['mean'] #+ [f"{i:02d}" for i in range(0, 51, 5)]

    # abm_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ABM/rmm.74toRealtime.txt"
    gt_filepath = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt'

    output_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/plots/phase_space'
    trajectory_length = 28

    os.makedirs(output_dir, exist_ok=True)

    gt_df = load_rmm_indices(gt_filepath)
    # abm_df = load_rmm_indices(abm_filepath)
    

    pred_composites_per_dir = []
    gt_composites_per_dir = []

    for dir in predict_dirs:
        pred_composites, gt_composites = compute_composite_by_phase(dir, gt_df, member=None, trajectory_length=trajectory_length, window=smoothing_window, start_date=start_date)
        pred_composites_per_dir.append(pred_composites)
        gt_composites_per_dir.append(gt_composites)
    
    ensemble_member_labels = []
    for i, dir in enumerate(ensemble_dirs):
        for member in ensemble_members:
            pred_composites, gt_composites = compute_composite_by_phase(dir, gt_df, member=member, trajectory_length=trajectory_length, window=smoothing_window, start_date=start_date)
            pred_composites_per_dir.append(pred_composites)
            gt_composites_per_dir.append(gt_composites)
            ensemble_member_labels.append(f'{ensemble_labels[i]} {member}')


    output_filename = 'phase_space_composite.png'
    phase_space_composite_plot(
        pred_composites=pred_composites_per_dir,
        gt_composites=gt_composites_per_dir[0],
        labels=labels + ensemble_member_labels,
        gt_label='Ours',
        title=f'{trajectory_length}-day trajectory phase space composite diagram',
        output_filename=os.path.join(output_dir, output_filename)
    )

if __name__ == "__main__":
    main()
