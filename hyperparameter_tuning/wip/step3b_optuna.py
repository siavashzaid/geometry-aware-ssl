"""
Step 3b: Final Training on 500k
Run this script directly:  python step3b.py
"""

import sys
sys.path.append("../src")
from precomputed_dataset import precomputedDataset

from utils import (
    SEED, RESULTS_DIR,
    TRAIN_PATH_STEP3B, VAL_PATH, TEST_PATH,
    get_device, build_model, evaluate, train,
)

import json
import numpy as np
import torch
from pathlib import Path


# =============================================================================
# Step 3b Main
# =============================================================================

def run_step3b():
    """Final training with the fully-specified best config on 500k samples.

    This is the ONLY step that touches the test set. All previous steps used
    only train and validation data. This ensures the test metric is an honest
    estimate of generalization to unseen geometries and source positions.

    Trains from scratch (not continuing from any checkpoint) because:
    1. Cleaner and more defensible for a thesis
    2. The 500k loss landscape may have different (better) basins than smaller scales
    3. The computational cost difference of warm-starting is modest

    Saves both the final results JSON and a PyTorch checkpoint (.pt) containing
    the model weights, config, and test results -- everything needed to reload
    the model for deployment, further evaluation, or visualization.
    """
    device = get_device()
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 3b: Final Training on 500k")
    print("=" * 70 + "\n")

    # Load the fully-specified best config from Step 3a.
    best_config_path = results_dir / "step3a_best_config.json"
    if not best_config_path.exists():
        raise FileNotFoundError(
            f"Step 3a results not found at {best_config_path}. Run Step 3a first."
        )

    with open(best_config_path) as f:
        step3a_output = json.load(f)

    config = step3a_output["config"]

    # Step 3b training params: generous budget for final training.
    # 300 epochs max (but early stopping will likely trigger earlier).
    # Patience 50: very patient -- this is the final run, don't stop prematurely.
    config["max_epochs"] = 300
    config["early_stop_patience"] = 50

    # Print the complete final configuration for the record.
    print("Final configuration:")
    print("-" * 40)
    for k, v in sorted(config.items()):
        # Skip verbose scheduler sub-params for cleaner output.
        if not k.startswith("scheduler_") and k not in ("scheduler_mode", "scheduler_threshold_mode"):
            print(f"  {k}: {v}")
    print()

    # ---- Train ----
    # trial=None: no Optuna, no pruning. Just a single full training run.
    result, best_model_state = train(
        config=config,
        device=device,
        train_path=TRAIN_PATH_STEP3B,
        val_path=VAL_PATH,
        seed=SEED,
        trial=None,
    )

    # ==================== FINAL TEST EVALUATION ====================
    # This is the ONLY time we evaluate on the test set in the entire pipeline.
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    # Load the test dataset (no DataLoader needed -- sample-by-sample evaluation).
    test_ds = precomputedDataset(TEST_PATH)

    # Rebuild the model architecture and load the best weights.
    sample0 = test_ds[0]
    model = build_model(config, sample0, device)

    # Move the saved CPU state dict back to GPU before loading.
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Compute test metric using the same sample-by-sample evaluate() as training.
    test_loss_loc, test_loss_str, test_metric = evaluate(model, test_ds, device)

    # ---- Detailed Predictions for Accuracy Bands ----
    model.eval()
    all_preds = []
    all_targets = []
    all_str_preds = []
    all_str_targets = []

    with torch.no_grad():
        for data in test_ds:
            data = data.to(device)
            pred_loc, pred_str = model.forward_from_data(data)

            # Collect on CPU as numpy for analysis.
            all_preds.append(pred_loc.cpu().numpy().squeeze())
            all_targets.append(data.y.squeeze(0).cpu().numpy().squeeze())
            all_str_preds.append(pred_str.cpu().numpy().squeeze())
            all_str_targets.append(data.strength.squeeze(0).cpu().numpy().squeeze())

    # Stack into arrays: [N_test, 2] for locations, [N_test] for strengths.
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    str_preds = np.array(all_str_preds)
    str_targets = np.array(all_str_targets)

    # Euclidean distance errors between predicted and true source locations.
    errors = np.linalg.norm(preds - targets, axis=-1)  # [N_test]

    # Mean absolute error on strength (more interpretable than MSE for strengths).
    str_mae = np.abs(str_preds - str_targets).mean()

    # ---- Print Final Report ----
    print(f"\n  Test metric:    {test_metric:.6f}")
    print(f"  Test loss_loc:  {test_loss_loc:.6f}")
    print(f"  Test loss_str:  {test_loss_str:.6f}")
    print(f"  Strength MAE:   {str_mae:.4f}")

    # Accuracy bands: what fraction of predictions fall within X meters of the true location.
    print(f"\n  Accuracy Bands:")
    print(f"    < 0.05m: {(errors < 0.05).mean() * 100:.1f}%")
    print(f"    < 0.10m: {(errors < 0.10).mean() * 100:.1f}%")
    print(f"    < 0.50m: {(errors < 0.50).mean() * 100:.1f}%")
    print(f"    < 1.00m: {(errors < 1.00).mean() * 100:.1f}%")
    print(f"\n  Epochs trained: {result['epochs_trained']}")
    print(f"  Best epoch:     {result['best_epoch']}")
    print(f"  Parameters:     {result['n_params']:,}")
    print(f"  Runtime:        {result['runtime_seconds']/60:.1f} min")

    # ---- Save Final Results as JSON ----
    final_output = {
        "config": config,
        "test_metric": float(test_metric),
        "test_loss_loc": float(test_loss_loc),
        "test_loss_str": float(test_loss_str),
        "test_str_mae": float(str_mae),
        "accuracy_bands": {
            "0.05m": float((errors < 0.05).mean()),
            "0.10m": float((errors < 0.10).mean()),
            "0.50m": float((errors < 0.50).mean()),
            "1.00m": float((errors < 1.00).mean()),
        },
        "best_epoch": result["best_epoch"],
        "epochs_trained": result["epochs_trained"],
        "n_params": result["n_params"],
        "runtime_seconds": result["runtime_seconds"],
    }

    output_path = results_dir / "step3b_final_results.json"
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nFinal results saved to: {output_path}")

    # ---- Save Model Checkpoint ----
    checkpoint_path = results_dir / "step3b_best_model.pt"
    torch.save({
        "model_state_dict": best_model_state,
        "config": config,
        "test_results": final_output,
    }, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")

    # Cleanup
    del model, test_ds
    torch.cuda.empty_cache()

    return final_output


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_step3b()
