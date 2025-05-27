import os
import copy
import xarray as xr
from typing import Optional
import glob 
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from mjo.utils.transforms import NormalizeDenormalize
from mjo.utils.dataset import (
    Forecast,
    NPZReader,
    ShuffleIterableDataset,
    collate_fn,
)

class MJOForecastDataModule(LightningDataModule):
    """DataModule for MJO forecast data.

    Args:
        root_dir (str): Root directory for preprocessed data.
        in_variables (list): List of input variables.
        out_variables (list, optional): List of output variables.
        predictions (list, optional): List of predictions elements to include in output. 
            If provided must all be positive integers. Defaults to [] (current timestamp).
        history (list, optional): List of history elements to include in input. 
            If provided must all be negative integers. Defaults to [] (current timestamp).
        normalize_data (bool, optional): Flag to normalize data. Defaults to False.
        max_buffer_size (int): Maximum buffer size for shuffling. Defaults to 100.
        batch_size (int, optional): Batch size. Defaults to 64.
        num_workers (int, optional): Number of workers. Defaults to 0.
        pin_memory (bool, optional): Whether to pin memory. Defaults to False.
    """

    def __init__(
        self,
        root_dir: str,
        in_variables: list,
        out_variables: list,
        predictions: list = [],
        history: list = [],
        normalize_data: bool = False,
        max_buffer_size: int = 100,        
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        self.root_dir = root_dir
        self.in_variables = in_variables
        self.out_variables = out_variables
        self.predictions = predictions
        self.history = history
        self.normalize_data = normalize_data
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.train_file = os.path.join(self.root_dir, "train.npz")
        self.val_file = os.path.join(self.root_dir, "val.npz")
        self.test_file = os.path.join(self.root_dir, "test.npz")
        self.stats_file = os.path.join(self.root_dir, "statistics.npz")

        in_mean, in_std = self.get_normalization_stats(self.stats_file, self.in_variables)
        out_mean, out_std = self.get_normalization_stats(self.stats_file, self.out_variables)
        self.in_transforms = NormalizeDenormalize(in_mean, in_std)
        self.out_transforms = NormalizeDenormalize(out_mean, out_std)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

        if not self.normalize_data:
            print('Warning: Both input and output data will not be normalized. Model will be run on unnormalized data')

    def get_normalization_stats(self, file, variables):
        statistics = np.load(file)
        normalize_mean = np.array([statistics[f"{var}_mean"] for var in variables])
        normalize_std = np.array([statistics[f"{var}_std"] for var in variables])
        return normalize_mean, normalize_std
    
    def get_history(self):
        return self.history

    def get_predictions(self):
        return self.predictions
    
    def get_transforms(self, group: str):
        if group == 'in':
            return copy.deepcopy(self.in_transforms)
        elif group == 'out':
            return copy.deepcopy(self.out_transforms)
        else:
            raise ValueError(f"Invalid normalization group name: {group}")

    def update_normalization_stats(self, mean, std, group: str):
        if not self.normalize_data:
            print(f"Warning: Updating normalization statistics for normalization group: {group} when normalize_data is False. This will likely have no effect.")
        else:
            print(f"Info: Updating normalization statistics for normalization group: {group}")
        if group == 'in':
            self.in_transforms.update(mean, std)
        elif group == 'out':
            self.out_transforms.update(mean, std)
        else:
            raise ValueError(f"Invalid normalization group name: {group}")

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.data_train = ShuffleIterableDataset(
                Forecast(
                    NPZReader(
                        file_path=self.train_file,
                        in_variables=self.in_variables,
                        out_variables=self.out_variables,
                        predictions=self.predictions,
                        history=self.history,
                    ),
                    normalize_data = self.normalize_data,
                    in_transforms=self.in_transforms,
                    out_transforms=self.out_transforms,
                ),
                max_buffer_size=self.max_buffer_size,
            )
            self.data_val = Forecast(
                NPZReader(
                    file_path=self.val_file,
                    in_variables=self.in_variables,
                    out_variables=self.out_variables,
                    predictions=self.predictions,
                    history=self.history,
                ),
                normalize_data = self.normalize_data,
                in_transforms=self.in_transforms,
                out_transforms=self.out_transforms,
            )
                
        if stage == 'test':
            self.data_test = Forecast(
                NPZReader(
                    file_path=self.test_file,
                    in_variables=self.in_variables,
                    out_variables=self.out_variables,
                    predictions=self.predictions,
                    history=self.history,
                ),
                normalize_data = self.normalize_data,
                in_transforms=self.in_transforms,
                out_transforms=self.out_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

def main():

    # example usage
    root_dir = "/glade/derecho/scratch/kvirji/DATA/preprocessed/MJO/reference_period_1979-09-07_to_2001-12-31"
    in_variables = ["RMM1", "RMM2"]
    out_variables = ["RMM1"]
    predictions = [1, 2, 3, 4, 5]
    history = [-1, -2, -3, -4, -5]
    normalize_data = True
    max_buffer_size = 100
    batch_size = 4
    num_workers = 0
    pin_memory = False

    dm = MJOForecastDataModule(
        root_dir=root_dir,
        in_variables=in_variables,
        out_variables=out_variables,
        predictions=predictions,
        history=history,
        normalize_data=normalize_data,
        max_buffer_size=max_buffer_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()

    print("Iterating through training dataloader and printing samples:")
    for i, batch in enumerate(train_loader):
        print(batch)
        if i >= 10:
            break

if __name__ == "__main__":
    main()
