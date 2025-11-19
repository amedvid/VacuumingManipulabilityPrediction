# manipulability_model.py
import torch
import torch.nn as nn


class ManipulabilityMLP(nn.Module):
    """Fully-connected regression network 7 -> 6."""

    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        assert isinstance(hidden_dims, (list, tuple)) and len(hidden_dims) > 0, \
            "hidden_dims must be a non-empty list or tuple"

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_model(checkpoint_path, device=None):
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
    model = ManipulabilityMLP(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        hidden_dims=ckpt["hidden_dims"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model
