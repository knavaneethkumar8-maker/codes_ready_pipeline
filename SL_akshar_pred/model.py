
import torch
import torch.nn as nn

class SLNet(nn.Module):
    """A tiny classifier over 2 frames x 15 features -> class."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # x: (B,T,15) or (T,15)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b, t, f = x.shape
        if t < 2:
            pad = torch.zeros((b, 2 - t, f), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        x = x[:, :2, :].reshape(b, 30)
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs
