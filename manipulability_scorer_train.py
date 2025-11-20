# manipulability_scorer_train.py
from pathlib import Path
import time
import datetime as dt

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

from json_data_loader import create_dataloaders
from manipulability_model import ManipulabilityMLP


# -------------------------------
# Global config parameters
# -------------------------------
EPOCHS = 2000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DEVICE = "cuda"   # "cpu" if no GPU
VAL_RATIO = 0.2
HIDDEN_DIMS = (128, 256, 128, 64)
# -------------------------------

# Build run name with architecture + hyperparams + timestamp
HID_STR = "-".join(str(h) for h in HIDDEN_DIMS)
TIMESTAMP = dt.datetime.now().strftime("%Y_%m_%d_%H_%M")
RUN_NAME = f"MLP_E{EPOCHS}_BS{BATCH_SIZE}_LR{LEARNING_RATE}_H{HID_STR}_{TIMESTAMP}"

BASE_DIR = Path("experiments") / RUN_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = BASE_DIR / "model.pt"
LOG_DIR = BASE_DIR / "tb"


def evaluate(model, loader, device):
    """Compute MSE and MAE on validation set."""
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")

    total_mse = 0.0
    total_mae = 0.0
    total_n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            total_mse += mse_loss(y_pred, y).item()
            total_mae += mae_loss(y_pred, y).item()
            total_n += y.shape[0]

    return total_mse / total_n, total_mae / total_n


def train(data_path: str):
    """Main training loop."""
    start_time = time.perf_counter()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        data_path,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
    )

    # Print dataset sizes
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    total_size = train_size + val_size
    print(
        f"Loaded data from '{data_path}': "
        f"total samples = {total_size}, train = {train_size}, val = {val_size}"
    )

    model = ManipulabilityMLP(
        input_dim=7,
        output_dim=6,
        hidden_dims=HIDDEN_DIMS
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_mse": [], "val_mae": []}

    # TensorBoard writer
    writer = SummaryWriter(LOG_DIR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_mse, val_mae = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        # TensorBoard logging
        writer.add_scalar("Loss/train_mse", train_loss, epoch)
        writer.add_scalar("Loss/val_mse", val_mse, epoch)
        writer.add_scalar("Loss/val_mae", val_mae, epoch)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.6f}  "
            f"val_mse={val_mse:.6f}  "
            f"val_mae={val_mae:.6f}"
        )

    writer.close()

    # Save model
    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_dim": 7,
        "output_dim": 6,
        "hidden_dims": HIDDEN_DIMS,
        "history": history,
    }
    torch.save(ckpt, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

    # Plot curve
    out_png = BASE_DIR / "loss_curves.png"
    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(14, 6))
    plt.plot(epochs, history["train_loss"], label="Train MSE loss")
    plt.plot(epochs, history["val_mse"], label="Val MSE")
    plt.plot(epochs, history["val_mae"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Training curve saved to {out_png}")

    end_time = time.perf_counter()
    elapsed_sec = end_time - start_time
    elapsed_min = elapsed_sec / 60.0
    print(f"Training finished in {elapsed_sec:.2f} seconds (~{elapsed_min:.2f} minutes)")


train("data/vacuum_states_with_manipulability_scores_jinx.json")
