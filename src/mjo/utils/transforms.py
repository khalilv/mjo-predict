import torch

class NormalizeDenormalize:
    """Normalize and denormalize data using z-score normalization.

    Args:
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        device: PyTorch device (default: CPU)
        dtype: PyTorch dtype (default: float32)
    """
    def __init__(self, mean, std, device=torch.device('cpu'), dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.mean = torch.tensor(mean, device=self.device, dtype=self.dtype)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype)

    def to(self, device=None, dtype=None):
        """Move tensors to specified device and/or dtype.

        Args:
            device: Target device (optional)
            dtype: Target dtype (optional)
        """
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        self.mean = self.mean.to(device=self.device, dtype=self.dtype)
        self.std = self.std.to(device=self.device, dtype=self.dtype)
        return

    def normalize(self, x: torch.Tensor):
        """Normalize input using z-score normalization: (x - mean) / std.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        return (x - self.mean) / self.std

    def update(self, mean, std):
        """Update normalization parameters.

        Args:
            mean: New mean values
            std: New standard deviation values
        """
        assert len(mean) == len(self.mean), f"Length of mean values ({len(mean)}) must match existing mean length ({len(self.mean)})"
        assert len(std) == len(self.std), f"Length of std values ({len(std)}) must match existing std length ({len(self.std)})"
        self.mean = torch.tensor(mean, device=self.device, dtype=self.dtype)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype)

    def denormalize(self, x: torch.Tensor):
        """Denormalize input: x * std + mean.

        Args:
            x: Normalized tensor

        Returns:
            Denormalized tensor
        """
        return x * self.std + self.mean
