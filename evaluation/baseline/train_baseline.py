import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from evaluation.models.baseline import EigmodeTransformer
from src.datasets.baseline_dataset import baselineDataset

OUTPUT_DIR      = '/mnt/data/zaid/projects/results/training_baseline'
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'best_model.pt')

config = {
    "train_path":           "/mnt/data/zaid/projects/simulated_data/single_geometry_train.h5",
    "val_path":             "/mnt/data/zaid/projects/simulated_data/single_geometry_val.h5",
    "nchannels":            64,
    "num_layers":           12,
    "num_heads":            12,
    "dropout":              0.1,
    "lr":                   5e-4,
    "weight_decay":         1e-4,
    "batch_size":           256,
    "epochs":               200,
    "seed":                 0,
    "device":               "cuda:0",
}

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(config["seed"])
    device = config["device"]

    # --- data ---
    train_loader = DataLoader(baselineDataset(config["train_path"]), batch_size=config["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(baselineDataset(config["val_path"]),   batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # --- model ---
    model     = EigmodeTransformer(nchannels=config["nchannels"], num_layers=config["num_layers"], num_heads=config["num_heads"], dropout_rate=config["dropout"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], betas=(0.9, 0.999))

    best_val_loss, patience_counter, history = float("inf"), 0, []

    for epoch in range(1, config["epochs"] + 1):

        # --- train ---
        model.train()
        train_loss = 0.0
        for eigmode, loc, strength in train_loader:
            eigmode, loc, strength = eigmode.to(device), loc.to(device), strength.to(device)
            pred_str, pred_loc = model(eigmode)
            loss = F.mse_loss(pred_loc, loc) + F.mse_loss(pred_str, strength) 
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for eigmode, loc, strength in val_loader:
                eigmode, loc, strength = eigmode.to(device), loc.to(device), strength.to(device)
                pred_str, pred_loc = model(eigmode)
                val_loss += (F.mse_loss(pred_loc, loc) + F.mse_loss(pred_str, strength)).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    print(f"Done. Best val loss: {best_val_loss:.6f}")
