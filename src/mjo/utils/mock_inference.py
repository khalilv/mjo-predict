import os
from src.mjo.utils.RMM.io import load_rmm_indices, save_rmm_indices

predict_filepath = '/glade/derecho/scratch/kvirji/DATA/MJO/RMM/reference_period_1979-09-07_to_2001-12-31/rmm.txt'
output_dir = '/glade/derecho/scratch/kvirji/mjo-predict/exps/mock'
predict_length = 60

os.makedirs(output_dir, exist_ok=True)
predict_df = load_rmm_indices(predict_filepath, 1979)
slices = [predict_df.iloc[i:i+60] for i in range(0, len(predict_df)-60)]

#mock inference
for slice in slices:
    if slice.isna().any().any():
        continue
    filename = f'{str(slice.index.values[0]).split("T")[0]}.txt'
    save_rmm_indices(
        slice.index.values,
        slice.RMM1.values,
        slice.RMM2.values,
        os.path.join(output_dir, filename),
        'Mock'
    )
