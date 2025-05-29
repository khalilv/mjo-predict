# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torchmetrics import Metric

class MSE(Metric):
    def __init__(self, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms
        self.suffix = suffix
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape, f"Found shape mismatch between preds: {preds.shape} and targets: {targets.shape}"
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets)
            
        # preds, targets: (B, T, V)
        error = preds - targets
        squared_error = error ** 2
        self.sum_squared_error += squared_error.sum()
        self.num_elements += squared_error.numel()

    def compute(self):
        mse = self.sum_squared_error / self.num_elements
        name = f"mse_{self.suffix}" if self.suffix else "mse"
        return {name: mse}

class MAE(Metric):
    def __init__(self, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms
        self.suffix = suffix
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape, f"Found shape mismatch between preds: {preds.shape} and targets: {targets.shape}"
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets)

        # preds, targets: (B, T, V)
        abs_error = torch.abs(preds - targets)
        self.sum_abs_error += abs_error.sum()
        self.num_elements += abs_error.numel()

    def compute(self):
        mae = self.sum_abs_error / self.num_elements
        name = f"mae_{self.suffix}" if self.suffix else "mae"
        return {name: mae}