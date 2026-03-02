from src.training import build_model, evaluate_fn
from src.datasets.precomputed_dataset import precomputedDataset
from configs.final_config import final_config

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

MODEL_PATH = '/mnt/data/zaid/projects/results/training/best_model.pt'
TEST_PATH = '/mnt/data/zaid/projects/simulated_data/multigeometry_test.h5'

if __name__ == "__main__":
    device = final_config["device"]

    # --- Load model --- #
    model = build_model(final_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # --- Load test set --- #
    test_ds = precomputedDataset(TEST_PATH)
    test_loader = PyGDataLoader(test_ds, batch_size=final_config["val_batch_size"], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    # --- Evaluate on test set --- #
    test_loss, test_loss_loc, test_loss_str = evaluate_fn(model, test_loader, device)

    print(f"Test Loss: {test_loss:.6f}")
    print(f"  Test Loss (Localization): {test_loss_loc:.6f}")
    print(f"  Test Loss (Strength): {test_loss_str:.6f}")
