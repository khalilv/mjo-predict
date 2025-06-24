# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
import os
import torch
import numpy as np
from typing import Any, Optional
from pytorch_lightning import LightningModule
from mjo.TSMixer.model import TSMixerX
from mjo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from mjo.utils.metrics import MSE, MAE
from mjo.utils.RMM.io import save_rmm_indices

class MJOForecastModule(LightningModule):
    """
    PyTorch Lightning module for MJO (Madden-Julian Oscillation) forecasting using the TSMixer model.

    This module encapsulates the TSMixer architecture and training logic for MJO forecasting tasks.
    It supports flexible configuration of model architecture, optimization, and training schedule.

    Args:
        pretrained_path (str): Path to pre-trained checkpoint. Default: "".
        past_cov_dim (int): Number of past covariate features. Default: 0.
        future_cov_dim (int): Number of future covariate features. Default: 0.
        static_cov_dim (int): Number of static covariate features. Default: 0.
        nr_params (int): Number of parameters for the likelihood (or 1 if not probabilistic). Default: 1.
        hidden_size (int): Hidden state size of the TSMixer. Default: 64.
        ff_size (int): Dimension of the feedforward network internal to the module. Default: 64.
        num_blocks (int): Number of mixer blocks. Default: 2.
        activation (str): Activation function to use. Default: 'ReLU'.
        dropout (float): Dropout rate for regularization. Default: 0.1.
        norm_type (str): Type of normalization to use. Default: 'TimeBatchNorm2d'.
        normalize_before (bool): Whether to apply normalization before or after mixing. Default: True.
        lr (float): Learning rate. Default: 5e-4.
        beta_1 (float): Beta 1 for AdamW optimizer. Default: 0.9.
        beta_2 (float): Beta 2 for AdamW optimizer. Default: 0.99.
        weight_decay (float): Weight decay for AdamW optimizer. Default: 1e-5.
        warmup_steps (int): Number of warmup steps for learning rate scheduler. Default: 1000.
        max_steps (int): Maximum number of steps for learning rate scheduler. Default: 50000.
        save_outputs (bool): Whether to save model outputs. Default: False.
    """

    def __init__(
        self,
        pretrained_path: str = "",      
        past_cov_dim: int = 0,
        future_cov_dim: int = 0,
        static_cov_dim: int = 0,
        nr_params: int = 1,
        hidden_size: int = 64,
        ff_size: int = 64,
        num_blocks: int = 2,
        activation: str = 'ReLU',
        dropout: float = 0.1,
        norm_type: str = 'TimeBatchNorm2d',
        normalize_before: bool = True,
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        save_outputs: bool = False
    ):
        super().__init__()
        # Model architecture parameters
        self.pretrained_path = pretrained_path
        self.past_cov_dim = past_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        self.num_blocks = num_blocks
        self.activation = activation
        self.dropout = dropout
        self.norm_type = norm_type
        self.normalize_before = normalize_before

        # Optimization and scheduler parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        # Output and utility attributes
        self.save_outputs = save_outputs
        self.denormalization = None
        self.year_normalization = None

        self.save_hyperparameters(logger=False, ignore=["net"])      

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location=torch.device("cpu"), weights_only=True)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"), weights_only=True)
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=True)
        print(msg)

    def set_input_length(self, input_length: int):
        self.input_length = input_length
    
    def set_output_length(self, output_length: int):
        self.output_length = output_length
    
    def set_input_dim(self, input_dim: int):
        self.input_dim = input_dim
    
    def set_output_dim(self, output_dim: int):
        self.output_dim = output_dim
    
    def set_out_variables(self, out_variables: list):
        self.out_variables = out_variables

    def init_metrics(self):
        denormalize = self.denormalization.denormalize if self.denormalization else None

        self.train_mse = MSE(vars=self.out_variables, transforms=None, suffix='norm')
        self.val_mse = MSE(vars=self.out_variables, transforms=denormalize, suffix=None)        
        self.test_mse = MSE(vars=self.out_variables, transforms=denormalize, suffix=None)

    def init_network(self):
        self.net = TSMixerX(
            input_length=self.input_length,
            output_length=self.output_length,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            past_cov_dim=self.past_cov_dim,
            future_cov_dim=self.future_cov_dim,
            static_cov_dim=self.static_cov_dim,
            nr_params=self.nr_params,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            num_blocks=self.num_blocks,
            activation=self.activation,
            dropout=self.dropout,
            norm_type=self.norm_type,
            normalize_before=self.normalize_before,
        )
        if hasattr(self, "pretrained_path") and self.pretrained_path and len(self.pretrained_path) > 0:
            self.load_pretrained_weights(self.pretrained_path)
            
    def set_denormalization(self, denormalization):
        self.denormalization = denormalization
    
    def set_year_normalization(self, normalization):
        self.year_normalization = normalization
    
    def setup(self, stage: str):
        if self.denormalization:
            self.denormalization.to(device=self.device, dtype=self.dtype)
        if self.year_normalization:
            self.year_normalization.to(device=self.device, dtype=self.dtype)
    
    def get_timestamp_encodings(self, timestamps):
        years = torch.tensor(timestamps.astype('datetime64[Y]').astype(int) + 1970, device=self.device, dtype=self.dtype)
        if self.year_normalization:
            years = self.year_normalization.normalize(years)

        start_of_year = timestamps.astype('datetime64[Y]')
        doy = (timestamps - start_of_year).astype('timedelta64[D]').astype(int)
        next_year = (start_of_year + np.timedelta64(1, 'Y')).astype('datetime64[D]')
        days_in_year = (next_year - start_of_year).astype(int)  # shape (B, L)

        # Angle for sin/cos embedding
        angle = 2 * np.pi * doy / days_in_year
        sin = np.sin(angle)
        cos = np.cos(angle)

        doy_encoding = torch.tensor(np.stack([sin, cos], axis=-1),  device=self.device, dtype=self.dtype)

        timestamp_encodings = torch.cat([years.unsqueeze(-1), doy_encoding], dim=-1)

        return timestamp_encodings
    
    def prep_input(self, in_data: torch.Tensor, forecast_data: Optional[torch.Tensor], in_timestamps: np.ndarray, out_timestamps: np.ndarray, forecast_timestamps: Optional[np.ndarray]):
        
        in_timestamp_encodings = self.get_timestamp_encodings(in_timestamps)
        out_timestamp_encodings = self.get_timestamp_encodings(out_timestamps)

        if forecast_data is not None:
            assert (forecast_timestamps == out_timestamps).all(), 'Found mismatch between forecast timestamps and out timestamps'
            forecast_data = forecast_data.permute(0, 2, 1, 3) # (B, T, E, V)
            forecast_future = forecast_data.flatten(2, 3) # (B, T, E*V) flatten ensemble members to seperate variables
            forecast_past = torch.zeros((forecast_future.shape[0], in_data.shape[1], forecast_future.shape[2]), device=self.device, dtype=self.dtype) #forecast is only known in the future so we set these to 0
            x_past =  torch.cat([in_data, in_timestamp_encodings, forecast_past], dim=-1)
            x_future = torch.cat([out_timestamp_encodings, forecast_future], dim=-1) 
        else:
            x_past =  torch.cat([in_data, in_timestamp_encodings], dim=-1) 
            x_future = out_timestamp_encodings
        
        x_static = None    
        return (x_past, x_future, x_static)

    def training_step(self, batch: Any, batch_idx: int):
        in_data, out_data, forecast_data, in_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps = batch

        x_in = self.prep_input(in_data, forecast_data, in_timestamps, out_timestamps, forecast_timestamps)
        
        pred_data = self.net.forward(x_in=x_in)
        pred_data = pred_data.squeeze(dim=-1)

        batch_loss = self.train_mse(preds=pred_data, targets=out_data)
       
        for key in batch_loss.keys():
            self.log(
                "train/" + key,
                batch_loss[key],
                prog_bar=True,
            )

        self.train_mse.reset()
        return batch_loss['mse_norm']
    
    def validation_step(self, batch: Any, batch_idx: int):
        in_data, out_data, forecast_data, in_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps = batch

        x_in = self.prep_input(in_data, forecast_data, in_timestamps, out_timestamps, forecast_timestamps)
        
        pred_data = self.net.forward(x_in=x_in)
        pred_data = pred_data.squeeze(dim=-1)

        self.val_mse.update(preds=pred_data, targets=out_data)
        return
        
    def on_validation_epoch_end(self):
        val_mse = self.val_mse.compute()
               
        #scalar metrics
        loss_dict = {**val_mse}
        for key in loss_dict.keys():
            self.log(
                "val/" + key,
                loss_dict[key],
                prog_bar=True,
                sync_dist=True
            )
        self.val_mse.reset()
        return loss_dict
    
    def on_test_epoch_start(self):
        if self.save_outputs:
            self.output_dir = f'{self.logger.log_dir}/outputs/'
            os.makedirs(self.output_dir, exist_ok=False)

    def test_step(self, batch: Any, batch_idx: int):
        in_data, out_data, forecast_data, in_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps = batch
       
        x_in = self.prep_input(in_data, forecast_data, in_timestamps, out_timestamps, forecast_timestamps)

        pred_data = self.net.forward(x_in=x_in)
        pred_data = pred_data.squeeze(dim=-1)

        self.test_mse.update(preds=pred_data, targets=out_data)
       
        if self.save_outputs:
            pred_data = self.denormalization.denormalize(pred_data)
            pred_data = pred_data.cpu().numpy()
            for b in range(pred_data.shape[0]):
                filename = f'{str(in_timestamps[b][-1]).split("T")[0]}.txt'
                save_rmm_indices(
                    time=out_timestamps[b],
                    RMM1=pred_data[b,:,out_variables.index('RMM1')],
                    RMM2=pred_data[b,:,out_variables.index('RMM2')],
                    filename=os.path.join(self.output_dir, filename),
                    method_str='TSMixer'
                )
        return
    
    def on_test_epoch_end(self):
        test_mse = self.test_mse.compute()
               
        #scalar metrics
        loss_dict = {**test_mse}
        for key in loss_dict.keys():
            self.log(
                "test/" + key,
                loss_dict[key],
                prog_bar=False,
                sync_dist=True
            )
        self.test_mse.reset()
        return loss_dict

    #optimizer definition - will be used to optimize the network
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta_1, self.beta_2),
            weight_decay=self.weight_decay,
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.warmup_steps,
            self.max_steps,
            warmup_start_lr=self.lr/10,
            eta_min=self.lr/10,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
