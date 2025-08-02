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
    
class PerLeadTimeLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, num_leads=42):
        """
        Initializes a separate LSTM model for each lead time.

        Args:
            input_dim (int): Input feature dimension per timestep (e.g., 2 for RMM1/RMM2)
            hidden_dim (int): Number of hidden units in the LSTM (default: 10)
            num_leads (int): Number of lead times (e.g., 42)
        """
        super().__init__()
        self.num_leads = num_leads
        self.models = nn.ModuleList([
            nn.Sequential(
                LSTMBlock(input_dim, hidden_dim, input_dim)
            ) for _ in range(num_leads)
        ])

    def forward(self, x):
        """
        Forward pass for the batch of sequences.

        Args:
            x (Tensor): Shape (B, T, input_dim)

        Returns:
            Tensor of shape (B, T, input_dim)
        """
        B, T, D = x.shape
        assert T == self.num_leads, f"Expected {self.num_leads} leads, got {T}"

        outputs = []
        for t in range(T):
            xt = x[:, t, :].unsqueeze(1)  # shape: (B, 1, input_dim)
            yt = self.models[t](xt)       # shape: (B, input_dim)
            outputs.append(yt)
        return torch.stack(outputs, dim=1)  # (B, T, input_dim)


class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (B, 1, input_dim)
        Returns: (B, output_dim)
        """
        B = x.size(0)
        h0 = torch.zeros(1, B, self.lstm.hidden_size, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(1, B, self.lstm.hidden_size, device=x.device, dtype=x.dtype)
        out, _ = self.lstm(x, (h0, c0))  # out: (B, 1, hidden_dim)
        out = out.squeeze(1)            # (B, hidden_dim)
        return self.fc(out)             # (B, output_dim)
