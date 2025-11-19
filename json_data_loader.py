# json_data_loader.py
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class JsonStatesDataset(Dataset):
    """Loads (7 -> 6) mapping from your JSON file."""

    def __init__(self, json_path):
        json_path = Path(json_path)
        with json_path.open("r") as f:
            data = json.load(f)

        states = data["States"]
        x_list = []
        y_list = []

        for k, v in states.items():
            in_vals = [float(x) for x in k.split()]
            out_vals = [float(x) for x in v.split()]
            if len(in_vals) != 7 or len(out_vals) != 6:
                # Simple guard against malformed records
                continue
            x_list.append(in_vals)
            y_list.append(out_vals)

        self.x = torch.tensor(x_list, dtype=torch.float32)
        self.y = torch.tensor(y_list, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_dataloaders(
    json_path,
    batch_size=256,
    val_ratio=0.1,
    num_workers=0,
    seed=42,
):
    """Creates train/val loaders via random split."""
    dataset = JsonStatesDataset(json_path)
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
