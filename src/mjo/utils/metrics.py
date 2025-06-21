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
    

class BMSE(Metric):
    def __init__(self, max_lead_time, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms
        self.suffix = suffix
        self.max_lead_time = max_lead_time
        self.add_state("sum_squared_error", default=torch.zeros(max_lead_time), dist_reduce_fx="sum")
        self.add_state("sum_squared_amplitude_error", default=torch.zeros(max_lead_time), dist_reduce_fx="sum")
        self.add_state("sum_phase_error", default=torch.zeros(max_lead_time), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape, f"Found shape mismatch between preds: {preds.shape} and targets: {targets.shape}"
        assert preds.shape[2] == 2, f'Expected 2 variables (RMM1, RMM2) for BMSE calculation but found {preds.shape[2]}'
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets)

        # preds, targets: (B, T, 2)
        error = preds - targets
        squared_error = error ** 2
        self.sum_squared_error += squared_error.sum(dim=(0, 2))  # shape (T,)
        self.num_elements += squared_error.shape[0]  # batch size

        pred_amplitude = torch.sqrt(torch.sum(preds**2, dim=-1))  # shape: (B, T)
        target_amplitude = torch.sqrt(torch.sum(targets**2, dim=-1))  
        amplitude_error = pred_amplitude - target_amplitude
        amplitude_squared_error = amplitude_error ** 2
        self.sum_squared_amplitude_error = torch.sum(amplitude_squared_error, dim=0)

        pred_phase = torch.arctan2(preds[:,:,1], preds[:,:,0])  # shape: (B, T)
        target_phase = torch.arctan2(targets[:,:,1], targets[:,:,0]) 
        phase_error = 2*pred_amplitude*target_amplitude*(1-torch.cos(pred_phase - target_phase))
        self.sum_phase_error = torch.sum(phase_error, dim=0)


    def compute(self):
        bmse_loss_dict = {}
        bmsea_loss_dict = {}
        bmsep_loss_dict = {}
        for lt in range(self.max_lead_time):
            bmse = self.sum_squared_error[lt] / self.num_elements
            bmsea = self.sum_squared_amplitude_error[lt] / self.num_elements
            bmsep = self.sum_phase_error[lt] / self.num_elements
            assert torch.isclose(bmse, bmsea + bmsep), f'Found mismatch between BMSE {bmse} and components BMSEa {bmsea}, BMSEp {bmsep}'
            bmse_lt_name = f"bmse_{lt+1}_{self.suffix}" if self.suffix else f"bmse_{lt+1}"
            bmsea_lt_name = f"bmsea_{lt+1}_{self.suffix}" if self.suffix else f"bmsea_{lt+1}"
            bmsep_lt_name = f"bmsep_{lt+1}_{self.suffix}" if self.suffix else f"bmsep_{lt+1}"
            bmse_loss_dict[bmse_lt_name] = bmse
            bmsea_loss_dict[bmsea_lt_name] = bmsea
            bmsep_loss_dict[bmsep_lt_name] = bmsep

        bmse_name = f"bmse_{self.suffix}" if self.suffix else "bmse"
        bmsea_name = f"bmsea_{self.suffix}" if self.suffix else "bmsea"
        bmsep_name = f"bmsep_{self.suffix}" if self.suffix else "bmsep"
        bmse_loss_dict[bmse_name] = torch.mean(torch.stack(list(bmse_loss_dict.values())))
        bmsea_loss_dict[bmsea_name] = torch.mean(torch.stack(list(bmsea_loss_dict.values())))
        bmsep_loss_dict[bmsep_name] = torch.mean(torch.stack(list(bmsep_loss_dict.values())))

        return {**bmse_loss_dict, **bmsea_loss_dict, **bmsep_loss_dict}