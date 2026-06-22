import torch
import torch.nn as nn


class NAFNetWrapper(nn.Module):
    """Minimal NAFNet-compatible wrapper.

    This is a lightweight placeholder architecture that is easy to replace
    with a full NAFNet implementation or pretrained weights later.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_weights(self, path: str) -> bool:
        """Load state dict from `path`. Returns True on success, False otherwise."""
        try:
            state = torch.load(path, map_location="cpu")
            # support both state_dict and full checkpoints
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.load_state_dict(state)
            return True
        except Exception as e:
            print("model load failed", e)
            return False
