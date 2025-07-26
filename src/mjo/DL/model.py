import torch
import torch.nn as nn

class PerLeadTimeMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_leads=42, depth=2):
        """
        A separate submodel (MLP) for each lead time, with configurable depth.

        Args:
            input_dim (int): Input feature dimension per timestep (e.g., 2 for RMM1/RMM2)
            hidden_dim (int): Hidden layer size for each submodel
            num_leads (int): Number of lead times (T)
            depth (int): Number of layers in each MLP (> 0)
        """
        super().__init__()
        assert depth >= 1, "Depth must be at least 1"

        self.num_leads = num_leads
        self.models = nn.ModuleList([
            self._build_mlp(input_dim, hidden_dim, input_dim, depth)
            for _ in range(num_leads)
        ])

    def _build_mlp(self, in_dim, hidden_dim, out_dim, depth):
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, input_dim)

        Returns:
            Tensor of shape (B, T, input_dim), where each timestep prediction
            is from its corresponding lead-time-specific model.
        """
        B, T, D = x.shape
        assert T == self.num_leads, f"Expected {self.num_leads} lead times, got {T}"

        preds = []
        for t in range(T):
            preds.append(self.models[t](x[:, t, :]))  # (B, input_dim)
        return torch.stack(preds, dim=1)  # (B, T, input_dim)