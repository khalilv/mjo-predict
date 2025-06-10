"""
Temporal Fusion Transformer (TFT)
-------
"""

from typing import Union, Tuple, Dict, List

import torch
from torch import nn
from torch.nn import LSTM as _LSTM

from mjo.TFT import activations
from mjo.utils import normalization

from mjo.TFT.submodules import (
    _GateAddNorm,
    _GatedResidualNetwork,
    _InterpretableMultiHeadAttention,
    _MultiEmbedding,
    _VariableSelectionNetwork,
)

GLU_FFN = ["GLU", "Bilinear", "ReGLU", "GEGLU", "SwiGLU", "ReLU", "GELU"]


class TFTModel(nn.Module):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_dim: Tuple[int, int],
        variables_meta: Dict[str, Dict[str, List[str]]],
        num_static_components: int,
        hidden_size: Union[int, List[int]],
        lstm_layers: int,
        num_attention_heads: int,
        full_attention: bool,
        feed_forward: str,
        hidden_continuous_size: int,
        categorical_embedding_sizes: Dict[str, Tuple[int, int]],
        dropout: float,
        add_relative_index: bool,
        norm_type: Union[str, nn.Module],
        **kwargs,
    ):
        """PyTorch module implementing the TFT architecture from `this paper <https://arxiv.org/pdf/1912.09363.pdf>`_
        The implementation is built upon `pytorch-forecasting's TemporalFusionTransformer
        <https://pytorch-forecasting.readthedocs.io/en/latest/models.html>`_.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
            Also called: Encoder length
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
            Also called: Decoder length
        output_dim : Tuple[int, int]
            shape of output given by (n_targets, loss_size). (loss_size corresponds to nr_params in other models).
        variables_meta : Dict[str, Dict[str, List[str]]]
            dict containing variable encoder, decoder variable names for mapping tensors in `_TFTModule.forward()`
        num_static_components
            the number of static components (not variables) of the input target series. This is either equal to the
            number of target components or 1.
        hidden_size : int
            hidden state size of the TFT. It is the main hyper-parameter and common across the internal TFT
            architecture.
        lstm_layers : int
            number of layers for the Long Short Term Memory (LSTM) Encoder and Decoder (1 is a good default).
        num_attention_heads : int
            number of attention heads (4 is a good default)
        full_attention : bool
            If `True`, applies multi-head attention query on past (encoder) and future (decoder) parts. Otherwise,
            only queries on future part. Defaults to `False`.
        feed_forward
            Set the feedforward network block. default `GatedResidualNetwork` or one of the  glu variant.
            Defaults to `GatedResidualNetwork`.
        hidden_continuous_size : int
            default for hidden size for processing continuous variables.
        categorical_embedding_sizes : dict
            A dictionary containing embedding sizes for categorical static covariates. The keys are the column names
            of the categorical static covariates. The values are tuples of integers with
            `(number of unique categories, embedding size)`. For example `{"some_column": (64, 8)}`.
            Note that `TorchForecastingModels` can only handle numeric data. Consider transforming/encoding your data
            with `darts.dataprocessing.transformers.static_covariates_transformer.StaticCovariatesTransformer`.
        dropout : float
            Fraction of neurons affected by Dropout.
        add_relative_index : bool
            Whether to add positional values to future covariates. Defaults to `False`.
            This allows to use the TFTModel without having to pass future_covariates to `fit()` and `train()`.
            It gives a value to the position of each step from input and output chunk relative to the prediction
            point. The values are normalized with `input_chunk_length`.
        likelihood
            The likelihood model to be used for probabilistic forecasts. By default, the TFT uses
            a ``QuantileRegression`` likelihood.
        norm_type: str | nn.Module
            The type of LayerNorm variant to use.
        **kwargs
            all parameters required for :class:`darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
            base class.
        """

        super().__init__(**kwargs)

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_targets, self.loss_size = output_dim
        self.variables_meta = variables_meta
        self.num_static_components = num_static_components
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.categorical_embedding_sizes = categorical_embedding_sizes
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.full_attention = full_attention
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.add_relative_index = add_relative_index

        if isinstance(norm_type, str):
            try:
                self.layer_norm = getattr(normalization, norm_type)
            except AttributeError:
                raise AttributeError("Please provide a valid layer norm type")
        else:
            self.layer_norm = norm_type

        # initialize last batch size to check if new mask needs to be generated
        self.batch_size_last = -1
        self.attention_mask = None
        self.relative_index = None

        # general information on variable name endings:
        # _vsn: VariableSelectionNetwork
        # _grn: GatedResidualNetwork
        # _glu: GatedLinearUnit
        # _gan: GateAddNorm
        # _attn: Attention

        # # processing inputs
        # embeddings
        self.input_embeddings = _MultiEmbedding(
            embedding_sizes=categorical_embedding_sizes,
            variable_names=self.categorical_static_variables,
        )

        # continuous variable processing
        self.prescalers_linear = {
            name: nn.Linear(
                (
                    1
                    if name not in self.numeric_static_variables
                    else self.num_static_components
                ),
                self.hidden_continuous_size,
            )
            for name in self.reals
        }

        # static (categorical and numerical) variables
        static_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.categorical_static_variables
        }
        static_input_sizes.update({
            name: self.hidden_continuous_size for name in self.numeric_static_variables
        })

        self.static_covariates_vsn = _VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={
                name: True for name in self.categorical_static_variables
            },
            dropout=self.dropout,
            prescalers=self.prescalers_linear,
            single_variable_grns={},
            context_size=None,  # no context for static variables
            layer_norm=self.layer_norm,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.hidden_continuous_size for name in self.encoder_variables
        }

        decoder_input_sizes = {
            name: self.hidden_continuous_size for name in self.decoder_variables
        }

        self.encoder_vsn = _VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for non-static categorical inputs
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers_linear,
            single_variable_grns={},
            layer_norm=self.layer_norm,
        )

        self.decoder_vsn = _VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for non-static categorical inputs
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers_linear,
            single_variable_grns={},
            layer_norm=self.layer_norm,
        )

        # static encoders
        # for variable selection
        self.static_context_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # for hidden state of the lstm
        self.static_context_hidden_encoder_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # for cell state of the lstm
        self.static_context_cell_encoder_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = _LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = _LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # post lstm GateAddNorm
        self.post_lstm_gan = _GateAddNorm(
            input_size=self.hidden_size, dropout=dropout, layer_norm=self.layer_norm
        )

        # static enrichment and processing past LSTM
        self.static_enrichment_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size,
            layer_norm=self.layer_norm,
        )

        # attention for long-range processing
        self.multihead_attn = _InterpretableMultiHeadAttention(
            d_model=self.hidden_size,
            n_head=self.num_attention_heads,
            dropout=self.dropout,
        )
        self.post_attn_gan = _GateAddNorm(
            self.hidden_size, dropout=self.dropout, layer_norm=self.layer_norm
        )

        if self.feed_forward == "GatedResidualNetwork":
            self.feed_forward_block = _GatedResidualNetwork(
                self.hidden_size,
                self.hidden_size,
                self.hidden_size,
                dropout=self.dropout,
                layer_norm=self.layer_norm,
            )
        else:
            if self.feed_forward not in GLU_FFN:
                raise ValueError(
                    f"'{self.feed_forward}' is not in {GLU_FFN + ['GatedResidualNetwork']}"
                )
            # use glu variant feedforward layers
            # 4 is a commonly used feedforward multiplier
            self.feed_forward_block = getattr(activations, self.feed_forward)(
                d_model=self.hidden_size, d_ff=self.hidden_size * 4, dropout=dropout
            )

        # output processing -> no dropout at this late stage
        self.pre_output_gan = _GateAddNorm(
            self.hidden_size, dropout=None, layer_norm=self.layer_norm
        )

        self.output_layer = nn.Linear(self.hidden_size, self.n_targets * self.loss_size)

        self._attn_out_weights = None
        self._static_covariate_var = None
        self._encoder_sparse_weights = None
        self._decoder_sparse_weights = None

    @property
    def reals(self) -> List[str]:
        """
        List of all continuous variables in model
        """
        return self.variables_meta["model_config"]["reals_input"]

    @property
    def static_variables(self) -> List[str]:
        """
        List of all static variables in model
        """
        return self.variables_meta["model_config"]["static_input"]

    @property
    def numeric_static_variables(self) -> List[str]:
        """
        List of numeric static variables in model
        """
        return self.variables_meta["model_config"]["static_input_numeric"]

    @property
    def categorical_static_variables(self) -> List[str]:
        """
        List of categorical static variables in model
        """
        return self.variables_meta["model_config"]["static_input_categorical"]

    @property
    def encoder_variables(self) -> List[str]:
        """
        List of all encoder variables in model (excluding static variables)
        """
        return self.variables_meta["model_config"]["time_varying_encoder_input"]

    @property
    def decoder_variables(self) -> List[str]:
        """
        List of all decoder variables in model (excluding static variables)
        """
        return self.variables_meta["model_config"]["time_varying_decoder_input"]

    @staticmethod
    def expand_static_context(context: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, time_steps, -1)

    @staticmethod
    def get_relative_index(
        encoder_length: int,
        decoder_length: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns scaled time index relative to prediction point.
        """
        index = torch.arange(
            encoder_length + decoder_length, dtype=dtype, device=device
        )
        prediction_index = encoder_length - 1
        index[:encoder_length] = index[:encoder_length] / prediction_index
        index[encoder_length:] = index[encoder_length:] / prediction_index
        return index.reshape(1, len(index), 1).repeat(batch_size, 1, 1)

    @staticmethod
    def get_attention_mask_full(
        time_steps: int, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer.
        """
        eye = torch.eye(time_steps, dtype=dtype, device=device)
        mask = torch.cumsum(eye.unsqueeze(0).repeat(batch_size, 1, 1), dim=1)
        return mask < 1

    @staticmethod
    def get_attention_mask_future(
        encoder_length: int,
        decoder_length: int,
        batch_size: int,
        device: str,
        full_attention: bool,
    ) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer that acts on future input only.
        The model will attend to all `False` values.
        """
        if full_attention:
            # attend to entire past and future input
            decoder_mask = torch.zeros(
                (decoder_length, decoder_length), dtype=torch.bool, device=device
            )
        else:
            # attend only to past steps relative to forecasting step in the future
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = attend_step >= predict_step
        # attend to all past input
        encoder_mask = torch.zeros(
            batch_size, encoder_length, dtype=torch.bool, device=device
        )
        # combine masks along attended time - first encoder and then decoder

        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(batch_size, -1, -1),
            ),
            dim=2,
        )
        return mask

    def forward(self, x_in: tuple) -> torch.Tensor:
        """TFT model forward pass.

        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(n_samples, n_time_steps, n_variables)`

        Returns
        -------
        torch.Tensor
            the output tensor
        """
        x_cont_past, x_cont_future, x_static = x_in
        dim_samples, dim_time, dim_variable = 0, 1, 2
        device = x_in[0].device

        batch_size = x_cont_past.shape[dim_samples]
        encoder_length = self.input_chunk_length
        decoder_length = self.output_chunk_length
        time_steps = encoder_length + decoder_length

        # avoid unnecessary regeneration of attention mask
        if batch_size != self.batch_size_last:
            self.attention_mask = self.get_attention_mask_future(
                encoder_length=encoder_length,
                decoder_length=decoder_length,
                batch_size=batch_size,
                device=device,
                full_attention=self.full_attention,
            )
            if self.add_relative_index:
                self.relative_index = self.get_relative_index(
                    encoder_length=encoder_length,
                    decoder_length=decoder_length,
                    batch_size=batch_size,
                    device=device,
                    dtype=x_cont_past.dtype,
                )

            self.batch_size_last = batch_size

        if self.add_relative_index:
            x_cont_past = torch.cat(
                [
                    ts[:, :encoder_length, :]
                    for ts in [x_cont_past, self.relative_index]
                    if ts is not None
                ],
                dim=dim_variable,
            )
            x_cont_future = torch.cat(
                [
                    ts[:, -decoder_length:, :]
                    for ts in [x_cont_future, self.relative_index]
                    if ts is not None
                ],
                dim=dim_variable,
            )

        input_vectors_past = {
            name: x_cont_past[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.encoder_variables)
        }
        input_vectors_future = {
            name: x_cont_future[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.decoder_variables)
        }

        # Embedding and variable selection
        if self.static_variables:
            # categorical static covariate embeddings
            if self.categorical_static_variables:
                static_embedding = self.input_embeddings(
                    torch.cat(
                        [
                            x_static[:, :, idx]
                            for idx, name in enumerate(self.static_variables)
                            if name in self.categorical_static_variables
                        ],
                        dim=1,
                    ).int()
                )
            else:
                static_embedding = {}
            # add numerical static covariates
            static_embedding.update({
                name: x_static[:, :, idx]
                for idx, name in enumerate(self.static_variables)
                if name in self.numeric_static_variables
            })
            static_embedding, static_covariate_var = self.static_covariates_vsn(
                static_embedding
            )
        else:
            static_embedding = torch.zeros(
                (x_cont_past.shape[0], self.hidden_size),
                dtype=x_cont_past.dtype,
                device=device,
            )
            static_covariate_var = None

        static_context_expanded = self.expand_static_context(
            context=self.static_context_grn(static_embedding), time_steps=time_steps
        )

        embeddings_varying_encoder = {
            name: input_vectors_past[name] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_vsn(
            x=embeddings_varying_encoder,
            context=static_context_expanded[:, :encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors_future[name] for name in self.decoder_variables
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_vsn(
            x=embeddings_varying_decoder,
            context=static_context_expanded[:, encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = (
            self.static_context_hidden_encoder_grn(static_embedding)
            .expand(self.lstm_layers, -1, -1)
            .contiguous()
        )
        input_cell = (
            self.static_context_cell_encoder_grn(static_embedding)
            .expand(self.lstm_layers, -1, -1)
            .contiguous()
        )

        # run local lstm encoder
        encoder_out, (hidden, cell) = self.lstm_encoder(
            input=embeddings_varying_encoder, hx=(input_hidden, input_cell)
        )

        # run local lstm decoder
        decoder_out, _ = self.lstm_decoder(
            input=embeddings_varying_decoder, hx=(hidden, cell)
        )

        lstm_layer = torch.cat([encoder_out, decoder_out], dim=dim_time)
        input_embeddings = torch.cat(
            [embeddings_varying_encoder, embeddings_varying_decoder], dim=dim_time
        )

        # post lstm GateAddNorm
        lstm_out = self.post_lstm_gan(x=lstm_layer, skip=input_embeddings)

        # static enrichment
        static_context_enriched = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment_grn(
            x=lstm_out,
            context=self.expand_static_context(
                context=static_context_enriched, time_steps=time_steps
            ),
        )

        # multi-head attention
        attn_out, attn_out_weights = self.multihead_attn(
            q=attn_input[:, encoder_length:],
            k=attn_input,
            v=attn_input,
            mask=self.attention_mask,
        )

        # skip connection over attention
        attn_out = self.post_attn_gan(
            x=attn_out,
            skip=attn_input[:, encoder_length:],
        )

        # feed-forward
        out = self.feed_forward_block(x=attn_out)

        # skip connection over temporal fusion decoder from LSTM post _GateAddNorm
        out = self.pre_output_gan(
            x=out,
            skip=lstm_out[:, encoder_length:],
        )

        # generate output for n_targets and loss_size elements for loss evaluation
        out = self.output_layer(out)
        out = out.view(
            batch_size, self.output_chunk_length, self.n_targets, self.loss_size
        )
        self._attn_out_weights = attn_out_weights
        self._static_covariate_var = static_covariate_var
        self._encoder_sparse_weights = encoder_sparse_weights
        self._decoder_sparse_weights = decoder_sparse_weights
        return out

# minimal working example of tft
def main():

    input_chunk_length = 720
    output_chunk_length = 60
    output_dim = (2, 1)

    # variable metadata
    variables_meta = {
        "model_config": {
            "reals_input": ["RMM1", "RMM2", "doysin", "doycos", "year", "fuxiRMM1", "fuxiRMM2"],
            "time_varying_encoder_input": ["RMM1", "RMM2", "doysin", "doycos", "year", "fuxiRMM1", "fuxiRMM2"],
            "time_varying_decoder_input": ["doysin", "doycos", "year", "fuxiRMM1", "fuxiRMM2"],
            "static_input": [],
            "static_input_numeric": [],
            "static_input_categorical": [],
            "categorical_input": []
        }
    }

    # create model instance
    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        output_dim=output_dim,
        variables_meta=variables_meta,
        num_static_components=0,
        hidden_size=32,
        lstm_layers=1,
        num_attention_heads=4,
        full_attention=False,
        feed_forward="GatedResidualNetwork",
        hidden_continuous_size=16,
        categorical_embedding_sizes={},
        add_relative_index=True,
        dropout=0.6,
        norm_type="LayerNorm"
    )

    batch_size = 16
    x_cont_past = torch.randn(batch_size, input_chunk_length, 7) # past data (encoder input)
    x_cont_future = torch.randn(batch_size, output_chunk_length, 5) # future data (decoder input)

    # forward pass
    output = model((x_cont_past, x_cont_future, None)) #no static input in this example

    print(f"Input shapes:")
    print(f"Past data: {x_cont_past.shape}")
    print(f"Future data: {x_cont_future.shape}")
    print(f"\nOutput shape: {output.shape}")

    # example of accessing attention weights
    if model._attn_out_weights is not None:
        print(f"\nAttention weights shape: {model._attn_out_weights.shape}")

if __name__ == "__main__":
    main()