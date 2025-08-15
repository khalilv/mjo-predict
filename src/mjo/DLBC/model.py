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
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=2, num_leads=42):
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
                LSTMBlock(input_dim, hidden_dim, output_dim)
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
        return torch.concat(outputs, dim=1)  # (B, T, input_dim)


class LSTMBlock(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=2):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len=1, input_size=2)
        lstm_out, _ = self.lstm(x)         # (batch, 1, hidden)
        x = self.fc1(lstm_out)             # (batch, 1, hidden)
        x = self.activation(x)
        x = self.fc2(x)                    # (batch, 1, output_size=2)
        return x
