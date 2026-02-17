"""
Step 2: Validate Tier 1 Ranking on 25k
Run this script directly:  python step2.py
"""

from utils import (
    SEED, TIER2_DEFAULTS, HARD_FIXED, RESULTS_DIR,
    TRAIN_PATH_STEP2, VAL_PATH,
    get_device, train,
)

import json
from pathlib import Path


# =============================================================================
# Step 2 Main
# =============================================================================

def run_step2():
    """Validate the top Step 1 configs on a larger dataset (25k samples).

    PURPOSE: Check whether the ranking from Step 1 (10-15k) is stable when we
    increase the dataset size to 25k. If the top config at 10k is still top at 25k,
    we have confidence in the architecture. If rankings shuffle, we pick the config
    that's most stable across both scales.

    This step does NOT search any new parameters. It simply retrains the top 3-5
    configs from Step 1 with Tier 2 parameters still at defaults, but on more data
    and for more epochs (200 vs 120), with no Optuna pruning.

    The single best config from this step is passed to Step 3a for regularization tuning.
    """
    device = get_device()
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 2: Validate Tier 1 Ranking on 25k")
    print("=" * 70 + "\n")

    # Load the top configs that Step 1 exported.
    top_configs_path = results_dir / "step1_top_configs.json"
    if not top_configs_path.exists():
        raise FileNotFoundError(
            f"Step 1 results not found at {top_configs_path}. Run Step 1 first."
        )

    with open(top_configs_path) as f:
        top_configs = json.load(f)

    print(f"Loaded {len(top_configs)} configs from Step 1\n")

    results = []

    for i, tier1_params in enumerate(top_configs):
        print(f"\n{'='*70}")
        print(f"Config {i+1}/{len(top_configs)} (Step 1 Trial {tier1_params['trial_number']})")
        print(f"  Step 1 metric: {tier1_params['metric']:.6f}")
        print(f"{'='*70}")

        # Reconstruct the full config from saved Tier 1 params + defaults.
        config = {}

        # Tier 1: from the saved JSON
        for key in ["lr", "mpnn_num_layers", "attn_num_layers", "mpnn_hidden_dim",
                     "token_dim", "attn_num_heads", "lambda_str", "pooling_strategy"]:
            config[key] = tier1_params[key]

        # Tier 2: still at defaults (not searched until Step 3a)
        config["weight_decay"] = TIER2_DEFAULTS["weight_decay"]
        config["mpnn_dropout"] = TIER2_DEFAULTS["dropout"]
        config["attn_dropout"] = TIER2_DEFAULTS["dropout"]
        config["head_dropout"] = TIER2_DEFAULTS["dropout"]
        config["mp_layer_norm"] = TIER2_DEFAULTS["mp_layer_norm"]

        # Hard-fixed params
        config.update(HARD_FIXED)

        # Step 2 training params: longer training than Step 1, more patient early stopping.
        config["max_epochs"] = 200
        config["early_stop_patience"] = 30

        # Train with trial=None (no Optuna pruning -- we want all configs to run fully).
        result, model_state = train(
            config=config,
            device=device,
            train_path=TRAIN_PATH_STEP2,
            val_path=VAL_PATH,
            seed=SEED,
            trial=None,  # No pruning in Step 2
        )

        # Tag the result with its Step 1 provenance for traceability.
        result["step1_trial"] = tier1_params["trial_number"]
        result["step1_metric"] = tier1_params["metric"]
        results.append(result)

        # Show how this config performed at 25k vs its Step 1 (10k) metric.
        print(
            f"  -> Step 2 metric: {result['best_val_metric']:.6f} "
            f"(loc={result['best_val_loss_loc']:.6f}, str={result['best_val_loss_str']:.6f}) "
            f"vs Step 1: {tier1_params['metric']:.6f}"
        )

    # ---- Summary: show new ranking vs Step 1 ranking ----
    print("\n" + "=" * 70)
    print("STEP 2 RESULTS")
    print("=" * 70)

    # Sort by Step 2 metric (best first).
    results.sort(key=lambda r: r["best_val_metric"])

    for i, r in enumerate(results):
        rank_change = ""
        step1_rank = next(
            j for j, tc in enumerate(top_configs)
            if tc["trial_number"] == r["step1_trial"]
        )
        if i != step1_rank:
            rank_change = f" (was #{step1_rank + 1} in Step 1)"
        print(
            f"  #{i+1}{rank_change} | "
            f"Trial {r['step1_trial']} | "
            f"Metric: {r['best_val_metric']:.6f} "
            f"(loc={r['best_val_loss_loc']:.6f}, str={r['best_val_loss_str']:.6f})"
        )

    # ---- Save the single best config for Step 3a ----
    best_result = results[0]
    best_config = best_result["config"]
    output = {
        "config": best_config,
        "step2_metric": best_result["best_val_metric"],
        "step2_val_loss_loc": best_result["best_val_loss_loc"],
        "step2_val_loss_str": best_result["best_val_loss_str"],
        "step1_trial": best_result["step1_trial"],
    }
    output_path = results_dir / "step2_best_config.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBest config saved to: {output_path}")

    # Also save all Step 2 results for analysis (without the bulky history dicts).
    all_results_path = results_dir / "step2_all_results.json"
    serializable_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "history"}
        sr["final_val_metric"] = r["history"]["val_metric"][-1] if r["history"]["val_metric"] else None
        serializable_results.append(sr)
    with open(all_results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"All results saved to: {all_results_path}")

    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_step2()
