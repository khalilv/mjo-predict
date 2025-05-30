# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torchmetrics import Metric

class MSE(Metric):
    def __init__(self, vars, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms
        self.suffix = suffix
        self.vars = vars
        self.add_state("sum_squared_error", default=torch.zeros(len(vars)), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape, f"Found shape mismatch between preds: {preds.shape} and targets: {targets.shape}"
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets)
            
        # preds, targets: (B, T, V)
        error = preds - targets
        squared_error = error ** 2
        print(squared_error.sum(), squared_error.numel())
        self.sum_squared_error += squared_error.sum(dim=(0,1))
        self.num_elements += squared_error.shape[0] * squared_error.shape[1]

    def compute(self):
        loss_dict = {}
        for i, var in enumerate(self.vars):
            var_mse = self.sum_squared_error[i] / self.num_elements
            var_name = f"mse_{var}_{self.suffix}" if self.suffix else f"mse_{var}"
            loss_dict[var_name] = var_mse
        
        name = f"mse_{self.suffix}" if self.suffix else "mse"
        loss_dict[name] = torch.mean(torch.stack(list(loss_dict.values())))
        return loss_dict

class MAE(Metric):
    def __init__(self, vars, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms
        self.suffix = suffix
        self.vars = vars
        self.add_state("sum_abs_error", default=torch.zeros(len(vars)), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape, f"Found shape mismatch between preds: {preds.shape} and targets: {targets.shape}"
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets)

        # preds, targets: (B, T, V)
        abs_error = torch.abs(preds - targets)  # shape (B, T, V)
        self.sum_abs_error += abs_error.sum(dim=(0, 1))  # per-variable sum
        self.num_elements += abs_error.shape[0] * abs_error.shape[1]  # B*T

    def compute(self):
        loss_dict = {}
        for i, var in enumerate(self.vars):
            var_mae = self.sum_abs_error[i] / self.num_elements
            var_name = f"mae_{var}_{self.suffix}" if self.suffix else f"mae_{var}"
            loss_dict[var_name] = var_mae

        name = f"mae_{self.suffix}" if self.suffix else "mae"
        loss_dict[name] = torch.mean(torch.stack(list(loss_dict.values())))
        return loss_dict