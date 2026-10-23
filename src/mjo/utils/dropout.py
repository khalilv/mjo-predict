import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MonteCarloDropout(nn.Dropout):
    """Dropout that can stay active at inference for Monte Carlo sampling (https://arxiv.org/pdf/1506.02142.pdf)."""

    # MC dropout is off by default, so this behaves like standard dropout (active only in training mode)
    _mc_dropout_enabled = False

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.mc_dropout_enabled, self.inplace)

    @property
    def mc_dropout_enabled(self) -> bool:
        """Return True if dropout should be applied."""
        return self._mc_dropout_enabled or self.training
