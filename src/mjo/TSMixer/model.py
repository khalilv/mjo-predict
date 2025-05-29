"""
Time-Series Mixer (TSMixer)
---------------------------
"""

# The inner layers (``nn.Modules``) and the ``TimeBatchNorm2d`` were provided by a PyTorch implementation
# of TSMixer: https://github.com/ditschuk/pytorch-tsmixer
#
# The License of pytorch-tsmixer v0.2.0 from https://github.com/ditschuk/pytorch-tsmixer/blob/main/LICENSE,
# accessed Thursday, March 21st, 2024:
# 'The MIT License
#
# Copyright 2023 Konstantin Ditschuneit
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the “Software”), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# '

from typing import Callable, Optional, Union

import torch
from torch import nn

from mjo.TSMixer import normalization
from mjo.TSMixer.dropout import MonteCarloDropout

ACTIVATIONS = [
    "ReLU",
    "RReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU",
]

NORMS = [
    "LayerNorm",
    "LayerNormNoBias",
    "TimeBatchNorm2d",
]


def _time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series Tensor to a feature Tensor."""
    return x.permute(0, 2, 1)


class TimeBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        """A batch normalization layer that normalizes over the last two dimensions of a Tensor."""
        super().__init__(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x` has shape (batch_size, time, features)
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input Tensor, but got {x.ndim}D Tensor instead."
            )
        # apply 2D batch norm over reshape input_data `(batch_size, 1, timepoints, features)`
        output = super().forward(x.unsqueeze(1))
        # reshape back to (batch_size, timepoints, features)
        return output.squeeze(1)


class _FeatureMixing(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        ff_size: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout: float,
        normalize_before: bool,
        norm_type: nn.Module,
    ) -> None:
        """A module for feature mixing with flexibility in normalization and activation based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This module provides options for batch normalization before or after mixing
        features, uses dropout for regularization, and allows for different activation
        functions.

        Parameters
        ----------
        sequence_length
            The length of the input sequences.
        input_dim
            The number of input channels to the module.
        output_dim
            The number of output channels from the module.
        ff_size
            The dimension of the feed-forward network internal to the module.
        activation
            The activation function used within the feed-forward network.
        dropout
            The dropout probability used for regularization.
        normalize_before
            A boolean indicating whether to apply normalization before
            the rest of the operations.
        norm_type
            The type of normalization to use.
        """
        super().__init__()

        self.projection = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )
        self.norm_before = (
            norm_type((sequence_length, input_dim))
            if normalize_before
            else nn.Identity()
        )
        self.fc1 = nn.Linear(input_dim, ff_size)
        self.activation = activation
        self.dropout1 = MonteCarloDropout(dropout)
        self.fc2 = nn.Linear(ff_size, output_dim)
        self.dropout2 = MonteCarloDropout(dropout)
        self.norm_after = (
            norm_type((sequence_length, output_dim))
            if not normalize_before
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.projection(x)
        x = self.norm_before(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x_proj + x
        x = self.norm_after(x)
        return x


class _TimeMixing(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        activation: Callable,
        dropout: float,
        normalize_before: bool,
        norm_type: nn.Module,
    ) -> None:
        """Applies a transformation over the time dimension of a sequence based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This module applies a linear transformation followed by an activation function
        and dropout over the sequence length of the input feature torch.Tensor after converting
        feature maps to the time dimension and then back.

        Parameters
        ----------
        sequence_length
            The length of the sequences to be transformed.
        input_dim
            The number of input channels to the module.
        activation
            The activation function to be used after the linear
            transformation.
        dropout
            The dropout probability to be used after the activation function.
        normalize_before
            Whether to apply normalization before or after feature mixing.
        norm_type
            The type of normalization to use.
        """
        super().__init__()
        self.normalize_before = normalize_before
        self.norm_before = (
            norm_type((sequence_length, input_dim))
            if normalize_before
            else nn.Identity()
        )
        self.activation = activation
        self.dropout = MonteCarloDropout(dropout)
        self.fc1 = nn.Linear(sequence_length, sequence_length)
        self.norm_after = (
            norm_type((sequence_length, input_dim))
            if not normalize_before
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # permute the feature dim with the time dim
        x_temp = self.norm_before(x)
        x_temp = _time_to_feature(x_temp)
        x_temp = self.activation(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        # permute back the time dim with the feature dim
        x_temp = x + _time_to_feature(x_temp)
        x_temp = self.norm_after(x_temp)
        return x_temp


class _ConditionalMixerLayer(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        static_cov_dim: int,
        ff_size: int,
        activation: Callable,
        dropout: float,
        normalize_before: bool,
        norm_type: nn.Module,
    ) -> None:
        """Conditional mix layer combining time and feature mixing with static context based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This module combines time mixing and conditional feature mixing, where the latter
        is influenced by static features. This allows the module to learn representations
        that are influenced by both dynamic and static features.

        Parameters
        ----------
        sequence_length
            The length of the input sequences.
        input_dim
            The number of input channels of the dynamic features.
        output_dim
            The number of output channels after feature mixing.
        static_cov_dim
            The number of channels in the static feature input.
        ff_size
            The inner dimension of the feedforward network used in feature mixing.
        activation
            The activation function used in both mixing operations.
        dropout
            The dropout probability used in both mixing operations.
        normalize_before
            Whether to apply normalization before or after mixing.
        norm_type
            The type of normalization to use.
        """
        super().__init__()

        mixing_input = input_dim
        if static_cov_dim != 0:
            self.feature_mixing_static = _FeatureMixing(
                sequence_length=sequence_length,
                input_dim=static_cov_dim,
                output_dim=output_dim,
                ff_size=ff_size,
                activation=activation,
                dropout=dropout,
                normalize_before=normalize_before,
                norm_type=norm_type,
            )
            mixing_input += output_dim
        else:
            self.feature_mixing_static = None

        self.time_mixing = _TimeMixing(
            sequence_length=sequence_length,
            input_dim=mixing_input,
            activation=activation,
            dropout=dropout,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        self.feature_mixing = _FeatureMixing(
            sequence_length=sequence_length,
            input_dim=mixing_input,
            output_dim=output_dim,
            ff_size=ff_size,
            activation=activation,
            dropout=dropout,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(
        self, x: torch.Tensor, x_static: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.feature_mixing_static is not None:
            x_static_mixed = self.feature_mixing_static(x_static)
            x = torch.cat([x, x_static_mixed], dim=-1)
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x


class TSMixerX(nn.Module):
    def __init__(
        self,
        input_length: int, 
        output_length: int,
        input_dim: int,
        output_dim: int,
        past_cov_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        hidden_size: int,
        ff_size: int,
        num_blocks: int,
        activation: str,
        dropout: float,
        norm_type: Union[str, nn.Module],
        normalize_before: bool,
        **kwargs,
    ) -> None:
        """
        Initializes the TSMixer module for use within a Darts forecasting model.

        Parameters
        ----------
        input_length
            Historical length of input features.
        output_length
            Forecasting length of output features.
        input_dim
            Number of input target features.
        output_dim
            Number of output target features.
        past_cov_dim
            Number of past covariate features.
        future_cov_dim
            Number of future covariate features.
        static_cov_dim
            Number of static covariate features (number of target features
            (or 1 if global static covariates) * number of static covariate features).
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        hidden_size
           Hidden state size of the TSMixer.
        ff_size
            Dimension of the feedforward network internal to the module.
        num_blocks
            Number of mixer blocks.
        activation
            Activation function to use.
        dropout
            Dropout rate for regularization.
        norm_type
            Type of normalization to use.
        normalize_before
            Whether to apply normalization before or after mixing.
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.input_length = input_length
        self.output_length = output_length

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Invalid `activation={activation}`. Must be on of {ACTIVATIONS}."
            )
        activation = getattr(nn, activation)()

        if isinstance(norm_type, str):
            if norm_type not in NORMS:
                raise ValueError(
                    f"Invalid `norm_type={norm_type}`. Must be on of {NORMS}."
                )
            if norm_type == "TimeBatchNorm2d":
                norm_type = TimeBatchNorm2d
            else:
                norm_type = getattr(normalization, norm_type)
        else:
            norm_type = norm_type

        mixer_params = {
            "ff_size": ff_size,
            "activation": activation,
            "dropout": dropout,
            "norm_type": norm_type,
            "normalize_before": normalize_before,
        }

        self.fc_hist = nn.Linear(self.input_length, self.output_length)
        self.feature_mixing_hist = _FeatureMixing(
            sequence_length=self.output_length,
            input_dim=input_dim + past_cov_dim + future_cov_dim,
            output_dim=hidden_size,
            **mixer_params,
        )
        if future_cov_dim:
            self.feature_mixing_future = _FeatureMixing(
                sequence_length=self.output_length,
                input_dim=future_cov_dim,
                output_dim=hidden_size,
                **mixer_params,
            )
        else:
            self.feature_mixing_future = None
        self.conditional_mixer = self._build_mixer(
            prediction_length=self.output_length,
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            future_cov_dim=future_cov_dim,
            static_cov_dim=static_cov_dim,
            **mixer_params,
        )
        self.fc_out = nn.Linear(hidden_size, output_dim * nr_params)

    @staticmethod
    def _build_mixer(
        prediction_length: int,
        num_blocks: int,
        hidden_size: int,
        future_cov_dim: int,
        static_cov_dim: int,
        **kwargs,
    ) -> nn.ModuleList:
        """Build the mixer blocks for the model."""
        # the first block takes `x` consisting of concatenated features with size `hidden_size`:
        # - historic features
        # - optional future features
        input_dim_block = hidden_size * (1 + int(future_cov_dim > 0))

        mixer_layers = nn.ModuleList()
        for _ in range(num_blocks):
            layer = _ConditionalMixerLayer(
                input_dim=input_dim_block,
                output_dim=hidden_size,
                sequence_length=prediction_length,
                static_cov_dim=static_cov_dim,
                **kwargs,
            )
            mixer_layers.append(layer)
            # after the first block, `x` consists of previous block output with size `hidden_size`
            input_dim_block = hidden_size
        return mixer_layers

    def forward(self, x_in) -> torch.Tensor:
        # x_hist contains the historical time series data and the historical
        """TSMixer model forward pass.

        Parameters
        ----------
        x_in
            comes as Tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and
            `x_future` is the output/future chunk. Input dimensions are `(batch_size, time_steps,
            components)`.

        Returns
        -------
        torch.torch.Tensor
            The output  Tensorof shape `(batch_size, output_length, output_dim, nr_params)`.
        """
        # B: batch size
        # L: input chunk length
        # T: output chunk length
        # C: target components
        # P: past cov features
        # F: future cov features
        # S: static cov features
        # H = C + P + F: historic features
        # H_S: hidden Size
        # N_P: likelihood parameters

        # `x`: (B, L, H), `x_future`: (B, T, F), `x_static`: (B, C or 1, S)
        x, x_future, x_static = x_in

        # swap feature and time dimensions (B, L, H) -> (B, H, L)
        x = _time_to_feature(x)
        # linear transformations to horizon (B, H, L) -> (B, H, T)
        x = self.fc_hist(x)
        # (B, H, T) -> (B, T, H)
        x = _time_to_feature(x)

        # feature mixing for historical features (B, T, H) -> (B, T, H_S)
        x = self.feature_mixing_hist(x)
        if self.future_cov_dim:
            # feature mixing for future features (B, T, F) -> (B, T, H_S)
            x_future = self.feature_mixing_future(x_future)
            # (B, T, H_S) + (B, T, H_S) -> (B, T, 2*H_S)
            x = torch.cat([x, x_future], dim=-1)

        if self.static_cov_dim:
            # (B, C, S) -> (B, 1, C * S)
            x_static = x_static.reshape(x_static.shape[0], 1, -1)
            # repeat to match horizon (B, 1, C * S) -> (B, T, C * S)
            x_static = x_static.repeat(1, self.output_length, 1)

        for mixing_layer in self.conditional_mixer:
            # conditional mixer layers with static covariates (B, T, 2 * H_S), (B, T, C * S) -> (B, T, H_S)
            x = mixing_layer(x, x_static=x_static)

        # linear transformation to generate the forecast (B, T, H_S) -> (B, T, C * N_P)
        x = self.fc_out(x)
        # (B, T, C * N_P) -> (B, T, C, N_P)
        x = x.view(-1, self.output_length, self.output_dim, self.nr_params)
        return x
    

        # INSERT_YOUR_CODE
def main():
    # Example parameters for TSMixerX
    input_length = 10
    output_length = 4
    input_dim = 2
    output_dim = 2
    past_cov_dim = 1
    future_cov_dim = 0
    static_cov_dim = 0
    nr_params = 1
    hidden_size = 64
    ff_size = 64
    num_blocks = 8
    activation = "ReLU"
    dropout = 0.8
    norm_type = "TimeBatchNorm2d"
    normalize_before = True

    # Create dummy input data
    batch_size = 3
    # x_past: (B, L, H) where H = input_dim + past_cov_dim + future_cov_dim
    x_past = torch.randn(batch_size, input_length, input_dim + past_cov_dim + future_cov_dim)
    # x_future: (B, T, F)
    x_future = torch.randn(batch_size, output_length, future_cov_dim)
    # x_static: (B, C or 1, S)
    x_static = torch.randn(batch_size, output_dim, static_cov_dim)

    # Instantiate the model
    model = TSMixerX(
        input_length=input_length,
        output_length=output_length,
        input_dim=input_dim,
        output_dim=output_dim,
        past_cov_dim=past_cov_dim,
        future_cov_dim=future_cov_dim,
        static_cov_dim=static_cov_dim,
        nr_params=nr_params,
        hidden_size=hidden_size,
        ff_size=ff_size,
        num_blocks=num_blocks,
        activation=activation,
        dropout=dropout,
        norm_type=norm_type,
        normalize_before=normalize_before,
    )

    # Forward pass
    output = model((x_past, x_future, x_static))
    print("Output shape:", output.shape)
    print("Output:", output)

if __name__ == "__main__":
    main()