import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader as PyGDataLoader

from configs.final_config import final_config
from src.training import build_model, evaluate_fn
from src.datasets.precomputed_dataset import precomputedDataset

MODEL_PATH  = '/mnt/data/zaid/projects/results/baseline_multigeometry/best_model.pt'
TEST_PATH   = '/mnt/data/zaid/projects/simulated_data/randompos_test.h5'
OUTPUT_DIR  = '/mnt/data/zaid/projects/evaluation/baseline_multigeometry'

FILE_NAME = "test_predictions_randpos.csv"

DEVICE = "cuda:2"

def collect_predictions(model, loader, device):
    """
    Runs inference on loader and returns a DataFrame with
    predictions, ground truths, and errors.
    """
    model.eval()
    rows = []
    with torch.no_grad():
        for data in loader:
            # --- load to device ---
            data = data.to(device)

            # --- make predictions ---
            pred_loc, pred_str = model.forward_from_data(data)

            # --- squeeze to (B, 2) and (B,) ---
            pred_loc_np = pred_loc.squeeze().cpu().numpy().reshape(-1, 2)
            true_loc_np = data.y.squeeze().cpu().numpy().reshape(-1, 2)
            pred_str_np = pred_str.squeeze().cpu().numpy().reshape(-1)
            true_str_np = data.strength.squeeze().cpu().numpy().reshape(-1)

            # --- per-sample Euclidean localization error ---
            loc_error = np.linalg.norm(pred_loc_np - true_loc_np, axis=1)

            # --- collect results ---
            for i in range(len(loc_error)):
                rows.append({
                    "pred_x":      pred_loc_np[i, 0],
                    "pred_y":      pred_loc_np[i, 1],
                    "true_x":      true_loc_np[i, 0],
                    "true_y":      true_loc_np[i, 1],
                    "pred_str":    pred_str_np[i],
                    "true_str":    true_str_np[i],
                    "loc_error":   loc_error[i],
                    "str_error":   abs(pred_str_np[i] - true_str_np[i]),
                })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # --- Create output directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Set device ---
    device = DEVICE

    # --- Load model ---
    model = build_model(final_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # --- Load test set ---
    test_ds     = precomputedDataset(TEST_PATH)
    test_loader = PyGDataLoader(
        test_ds,
        batch_size=final_config["val_batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # --- Per-sample predictions ---
    df = collect_predictions(model, test_loader, device)

    csv_path = os.path.join(OUTPUT_DIR, FILE_NAME)
    df.to_csv(csv_path, index=False)

    print(f"\nPer-sample results saved to: {csv_path}")
    print(f"  Median loc error: {df['loc_error'].median():.4f}")
    print(f"  Mean   loc error: {df['loc_error'].mean():.4f}")