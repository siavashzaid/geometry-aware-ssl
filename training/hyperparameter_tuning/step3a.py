"""
Step 3a: Tier 2 Regularization Search on 100k
Run this script directly:  python step3a.py
"""

from utils import (
    SEED, HARD_FIXED, RESULTS_DIR,
    TRAIN_PATH_STEP3A, VAL_PATH,
    get_device, train,
)

import json
from pathlib import Path
from itertools import product


# =============================================================================
# Step 3a Main
# =============================================================================

def run_step3a():
    """Search Tier 2 regularization parameters on the locked Tier 1 architecture.

    PURPOSE: The Tier 1 architecture is now fixed (from Step 2). Here we find the
    optimal regularization settings (weight_decay, dropout, layer_norm) at a dataset
    scale (100k) that's closer to the final 500k training.

    Why 100k and not 500k? Because 18 runs at 500k is extremely expensive. 100k is a
    pragmatic intermediate scale where regularization dynamics are more representative
    of 500k than 25k, while still being affordable to grid-search.

    This uses an exhaustive GRID SEARCH (not Optuna) because:
    1. Only 3 parameters with small discrete spaces -> 3 x 3 x 2 = 18 combinations
    2. Grid search is simpler and more interpretable for a small space
    3. No need for Bayesian optimization when you can afford to try everything
    """
    device = get_device()
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 3a: Tier 2 Regularization Search on 100k")
    print("=" * 70 + "\n")

    # Load the best Tier 1 config from Step 2.
    best_config_path = results_dir / "step2_best_config.json"
    if not best_config_path.exists():
        raise FileNotFoundError(
            f"Step 2 results not found at {best_config_path}. Run Step 2 first."
        )

    with open(best_config_path) as f:
        step2_output = json.load(f)

    # base_config contains the full config (Tier 1 + Tier 2 defaults + hard-fixed)
    # from Step 2's winning trial. We'll override only the Tier 2 params.
    base_config = step2_output["config"]

    # Print the locked Tier 1 architecture for reference.
    print(f"Locked Tier 1 architecture (from Step 1 Trial {step2_output['step1_trial']}):")
    for key in ["lr", "mpnn_num_layers", "attn_num_layers", "mpnn_hidden_dim",
                 "token_dim", "attn_num_heads", "lambda_str", "pooling_strategy"]:
        print(f"  {key}: {base_config[key]}")
    print()

    # ---- Tier 2 Grid ----
    tier2_grid = {
        "weight_decay": [0.0, 1e-4, 1e-3],
        "dropout": [0.0, 0.05, 0.1],
        "mp_layer_norm": [True, False],
    }

    # Generate all 3 x 3 x 2 = 18 combinations using itertools.product.
    keys = list(tier2_grid.keys())
    combinations = list(product(*tier2_grid.values()))

    print(f"Tier 2 grid: {len(combinations)} combinations")
    print(f"  weight_decay: {tier2_grid['weight_decay']}")
    print(f"  dropout:      {tier2_grid['dropout']}")
    print(f"  mp_layer_norm: {tier2_grid['mp_layer_norm']}")
    print()

    results = []

    for idx, values in enumerate(combinations):
        # Convert (0.0, 0.05, True) -> {"weight_decay": 0.0, "dropout": 0.05, "mp_layer_norm": True}
        tier2_params = dict(zip(keys, values))

        print(f"\n{'─'*70}")
        print(f"  Run {idx+1}/{len(combinations)}: {tier2_params}")
        print(f"{'─'*70}")

        # Start from a COPY of the base config (don't mutate the original).
        config = dict(base_config)

        # Override the Tier 2 params with this grid point's values.
        # "dropout" is tied: same value for mpnn, attn, and head dropout.
        config["weight_decay"] = tier2_params["weight_decay"]
        config["mpnn_dropout"] = tier2_params["dropout"]
        config["attn_dropout"] = tier2_params["dropout"]
        config["head_dropout"] = tier2_params["dropout"]
        config["mp_layer_norm"] = tier2_params["mp_layer_norm"]

        # Step 3a training params: same as Step 2 (200 epochs, patience 30).
        config["max_epochs"] = 200
        config["early_stop_patience"] = 30

        # Train with trial=None (no Optuna, no pruning -- full grid search).
        result, model_state = train(
            config=config,
            device=device,
            train_path=TRAIN_PATH_STEP3A,
            val_path=VAL_PATH,
            seed=SEED,
            trial=None,
        )

        # Tag with the Tier 2 params used for easy summary printing.
        result["tier2_params"] = tier2_params
        results.append(result)

        print(
            f"  -> Metric: {result['best_val_metric']:.6f} "
            f"(loc={result['best_val_loss_loc']:.6f}, str={result['best_val_loss_str']:.6f}) "
            f"@ epoch {result['best_epoch']}"
        )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("STEP 3a RESULTS")
    print("=" * 70)

    # Sort by metric and display each combination's results.
    results.sort(key=lambda r: r["best_val_metric"])
    for i, r in enumerate(results):
        t2 = r["tier2_params"]
        print(
            f"  #{i+1} | "
            f"Metric: {r['best_val_metric']:.6f} "
            f"(loc={r['best_val_loss_loc']:.6f}, str={r['best_val_loss_str']:.6f}) | "
            f"wd={t2['weight_decay']}, drop={t2['dropout']}, ln={t2['mp_layer_norm']}"
        )

    # ---- Save the best FULL config (Tier 1 + Tier 2) for Step 3b ----
    best_result = results[0]
    best_config = best_result["config"]  # This config already has the winning Tier 2 values

    output = {
        "config": best_config,
        "step3a_metric": best_result["best_val_metric"],
        "step3a_val_loss_loc": best_result["best_val_loss_loc"],
        "step3a_val_loss_str": best_result["best_val_loss_str"],
        "tier2_params": best_result["tier2_params"],  # Which Tier 2 combo won
    }
    output_path = results_dir / "step3a_best_config.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBest full config saved to: {output_path}")

    # Save all 18 results for analysis.
    all_results_path = results_dir / "step3a_all_results.json"
    serializable_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "history"}
        serializable_results.append(sr)
    with open(all_results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"All results saved to: {all_results_path}")

    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_step3a()
