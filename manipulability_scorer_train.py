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
EPOCHS = 50
BATCH_SIZE = 128
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


def compute_target_stats(loader):
    """Compute mean and std for 6D targets over given loader (train set)."""
    sum_y = None
    sum_y2 = None
    n_samples = 0

    with torch.no_grad():
        for _, y in loader:
            y = y.to("cpu")
            if sum_y is None:
                sum_y = y.sum(dim=0, dtype=torch.float64)
                sum_y2 = (y * y).sum(dim=0, dtype=torch.float64)
            else:
                sum_y += y.sum(dim=0, dtype=torch.float64)
                sum_y2 += (y * y).sum(dim=0, dtype=torch.float64)
            n_samples += y.shape[0]

    mean = sum_y / n_samples
    var = sum_y2 / n_samples - mean * mean
    std = torch.sqrt(var + 1e-8)  # avoid zero std
    return mean.float(), std.float()


def evaluate(model, loader, device, target_mean, target_std):
    """
    Compute:
      - normalized MSE/MAE (in normalized space)
      - denormalized MSE/MAE (real values).
    """
    model.eval()
    mse_loss_norm = nn.MSELoss(reduction="sum")
    mae_loss_norm = nn.L1Loss(reduction="sum")

    total_mse_norm = 0.0
    total_mae_norm = 0.0
    total_mse_denorm = 0.0
    total_mae_denorm = 0.0
    total_n = 0

    tm = target_mean.to(device)
    ts = target_std.to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            # normalized
            y_norm = (y - tm) / ts
            y_pred_norm = (y_pred - tm) / ts

            total_mse_norm += mse_loss_norm(y_pred_norm, y_norm).item()
            total_mae_norm += mae_loss_norm(y_pred_norm, y_norm).item()

            # denormalized errors
            diff = y_pred - y
            total_mse_denorm += (diff * diff).sum().item()
            total_mae_denorm += diff.abs().sum().item()

            total_n += y.shape[0]

    mse_norm = total_mse_norm / total_n
    mae_norm = total_mae_norm / total_n
    mse_denorm = total_mse_denorm / total_n
    mae_denorm = total_mae_denorm / total_n

    return mse_norm, mae_norm, mse_denorm, mae_denorm


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

    # Compute target normalization stats on train set
    target_mean, target_std = compute_target_stats(train_loader)
    print("Target mean:", target_mean.tolist())
    print("Target std :", target_std.tolist())

    model = ManipulabilityMLP(
        input_dim=7,
        output_dim=6,
        hidden_dims=HIDDEN_DIMS
    ).to(device)

    criterion_norm = nn.MSELoss(reduction="sum")  # we will divide by N manually
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_mse_norm": [],
        "train_mse": [],
        "train_mae": [],
        "val_mse_norm": [],
        "val_mse": [],
        "val_mae": [],
    }

    # TensorBoard writer
    writer = SummaryWriter(LOG_DIR)

    tm = target_mean.to(device)
    ts = target_std.to(device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_mse_norm = 0.0
        total_mse_denorm = 0.0
        total_mae_denorm = 0.0
        total_n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)

            # normalized targets
            y_norm = (y - tm) / ts
            y_pred_norm = (y_pred - tm) / ts

            # loss in normalized space
            mse_norm_batch = criterion_norm(y_pred_norm, y_norm)  # sum over batch
            mse_norm_batch.backward()
            optimizer.step()

            bs = y.shape[0]
            total_mse_norm += mse_norm_batch.item()
            diff = y_pred - y
            total_mse_denorm += (diff * diff).sum().item()
            total_mae_denorm += diff.abs().sum().item()
            total_n += bs

        train_mse_norm = total_mse_norm / total_n
        train_mse = total_mse_denorm / total_n
        train_mae = total_mae_denorm / total_n

        (
            val_mse_norm,
            val_mae_norm,
            val_mse,
            val_mae,
        ) = evaluate(model, val_loader, device, target_mean, target_std)

        history["train_mse_norm"].append(train_mse_norm)
        history["train_mse"].append(train_mse)
        history["train_mae"].append(train_mae)
        history["val_mse_norm"].append(val_mse_norm)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        # TensorBoard logging: normalized and denormalized
        writer.add_scalar("Loss/train_mse_norm", train_mse_norm, epoch)
        writer.add_scalar("Loss/train_mse", train_mse, epoch)
        writer.add_scalar("Loss/train_mae", train_mae, epoch)

        writer.add_scalar("Loss/val_mse_norm", val_mse_norm, epoch)
        writer.add_scalar("Loss/val_mse", val_mse, epoch)
        writer.add_scalar("Loss/val_mae", val_mae, epoch)

        print(
            f"Epoch {epoch:04d}: "
            f"train_mse_norm={train_mse_norm:.6f}  "
            f"train_mse={train_mse:.6f}  "
            f"train_mae={train_mae:.6f}  "
            f"val_mse_norm={val_mse_norm:.6f}  "
            f"val_mse={val_mse:.6f}  "
            f"val_mae={val_mae:.6f}"
        )

    writer.close()

    # Save model + normalization stats
    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_dim": 7,
        "output_dim": 6,
        "hidden_dims": HIDDEN_DIMS,
        "history": history,
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
    }
    torch.save(ckpt, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

    # Plot curve (denormalized MSE/MAE – більш інтерпретовано)
    out_png = BASE_DIR / "loss_curves.png"
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(14, 6))
    plt.plot(epochs_range, history["train_mse"], label="Train MSE (denorm)")
    plt.plot(epochs_range, history["val_mse"], label="Val MSE (denorm)")
    plt.plot(epochs_range, history["val_mae"], label="Val MAE (denorm)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (original units)")
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
