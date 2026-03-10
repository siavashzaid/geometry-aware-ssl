import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Allow running from repo root or from baseline/
sys.path.insert(0, os.path.dirname(__file__))

from model import EigmodeTransformer
from dataset import baselineDataset

MODEL_PATH = '/mnt/data/zaid/projects/results/baseline_multigeometry/best_model.pt'
TEST_PATH  = '/mnt/data/zaid/projects/simulated_data/randompos_test.h5'
OUTPUT_DIR = '/mnt/data/zaid/projects/evaluation/baseline_multigeometry'

FILE_NAME = "test.csv"

DEVICE = "cuda:2"

config = {
    "nchannels":  64,
    "num_layers": 12,
    "num_heads":  8,
    "dropout":    0.1,
    "batch_size": 256,
    "device":     DEVICE,
}

def collect_predictions(model, loader, device):
    """
    Runs inference on loader and returns a DataFrame with
    predictions, ground truths, and per-sample errors.
    """
    model.eval()
    rows = []
    with torch.no_grad():
        for eigmode, loc, strength in loader:
            # --- load to device ---
            eigmode  = eigmode.to(device)
            loc      = loc.to(device)
            strength = strength.to(device)

            # --- make predictions ---
            pred_str, pred_loc = model(eigmode)

            # --- squeeze to (B, 2) and (B,) ---
            pred_loc_np = pred_loc.cpu().numpy().reshape(-1, 2)  # (B, 2)
            true_loc_np = loc.cpu().numpy().reshape(-1, 2)        # (B, 2)
            pred_str_np = pred_str.cpu().numpy().reshape(-1)      # (B,)
            true_str_np = strength.cpu().numpy().reshape(-1)      # (B,)

            # --- per-sample Euclidean localization error ---
            loc_error = np.linalg.norm(pred_loc_np - true_loc_np, axis=1)

            # --- collect results ---
            for i in range(len(loc_error)):
                rows.append({
                    "pred_x":    pred_loc_np[i, 0],
                    "pred_y":    pred_loc_np[i, 1],
                    "true_x":    true_loc_np[i, 0],
                    "true_y":    true_loc_np[i, 1],
                    "pred_str":  pred_str_np[i],
                    "true_str":  true_str_np[i],
                    "loc_error": loc_error[i],
                    "str_error": abs(pred_str_np[i] - true_str_np[i]),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # --- Create output directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Set device ---
    device = config["device"]

    # --- Load model ---
    model = EigmodeTransformer(
        nchannels=config["nchannels"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout_rate=config["dropout"],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # --- Load test set ---
    test_ds     = baselineDataset(TEST_PATH)
    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # --- Per-sample predictions ---
    df = collect_predictions(model, test_loader, device)

    csv_path = os.path.join(OUTPUT_DIR, FILE_NAME)
    df.to_csv(csv_path, index=False)

    print(f"Per-sample results saved to: {csv_path}")
    print(f"  Median loc error: {df['loc_error'].median():.4f}")
    print(f"  Mean   loc error: {df['loc_error'].mean():.4f}")
