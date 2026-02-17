"""
Step 1: Broad Tier 1 Search
Run this script directly:  python step1.py
"""

from utils import (
    LAMBDA_METRIC, SEED, TIER2_DEFAULTS, HARD_FIXED, RESULTS_DIR,
    TRAIN_PATH_STEP1, VAL_PATH, N_TRIALS_STEP1,
    get_device, train,
)

import json
import numpy as np
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


# =============================================================================
# Tier 1 Sampling
# =============================================================================

def sample_tier1(trial):
    """Sample Tier 1 hyperparameters for a single Optuna trial.

    Optuna's TPE sampler calls this function for each trial. 

    Args:
        trial: optuna.Trial object that provides the suggest_* sampling API.

    Returns:
        config: Complete flat dict with all hyperparameters needed by train().
    """
    config = {}

    # ---- Tier 1: Architecture Parameters (SEARCHED) ----
    config["lr"] = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    config["mpnn_num_layers"] = trial.suggest_categorical("mpnn_num_layers", [1, 2, 3, 4])
    config["attn_num_layers"] = trial.suggest_categorical("attn_num_layers", [2, 4, 6, 8])
    config["mpnn_hidden_dim"] = trial.suggest_categorical("mpnn_hidden_dim", [64, 128, 256])
    config["token_dim"] = trial.suggest_categorical("token_dim", [64, 128, 256])
    config["lambda_str"] = trial.suggest_categorical("lambda_str", [0.1, 0.3, 0.5, 1.0])
    config["pooling_strategy"] = trial.suggest_categorical("pooling_strategy", ["cls_token", "mean_pooling"])
    token_dim = config["token_dim"]
    valid_heads = [h for h in [2, 4, 8] if token_dim % h == 0]
    config["attn_num_heads"] = trial.suggest_categorical("attn_num_heads", valid_heads)

    # ---- Tier 2: Regularization Parameters (FIXED at defaults) ----
    config["weight_decay"] = TIER2_DEFAULTS["weight_decay"]
    config["mpnn_dropout"] = TIER2_DEFAULTS["dropout"]
    config["attn_dropout"] = TIER2_DEFAULTS["dropout"]
    config["head_dropout"] = TIER2_DEFAULTS["dropout"]
    config["mp_layer_norm"] = TIER2_DEFAULTS["mp_layer_norm"]

    # ---- Hard-Fixed Parameters ----
    config.update(HARD_FIXED)

    # ---- Step 1 Training Parameters ----
    config["max_epochs"] = 120
    config["early_stop_patience"] = 20

    return config


# =============================================================================
# Step 1 Main
# =============================================================================

def run_step1():
    """Execute Step 1: broad Tier 1 search using Optuna.

    Creates an Optuna study with TPE sampling and MedianPruner, then runs
    n_trials trials. Each trial samples Tier 1 hyperparameters, trains a model,
    and returns the best val_metric as the objective.
    """
    device = get_device()
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 1: Broad Tier 1 Search")
    print(f"  Trials: {N_TRIALS_STEP1}")
    print(f"  Data:   {TRAIN_PATH_STEP1}")
    print("=" * 70 + "\n")

    # Create the Optuna study.
    study = optuna.create_study(
        # We're minimizing val_metric 
        direction="minimize",

        # TPE (Tree-structured Parzen Estimator): Bayesian optimization sampler.
        # n_startup_trials=15: use pure random sampling for the first 15 trials to build a diverse baseline
        sampler=TPESampler(seed=SEED, n_startup_trials=15),

        # MedianPruner: early-stops trials that are below median performance.
        pruner=MedianPruner(
            n_startup_trials=15,
            n_warmup_steps=20,
            interval_steps=10,
        ),
    )

    def objective(trial):
        """Optuna objective function: sample config, train, return metric.

        This function is called once per trial by study.optimize().
        It must return a single float value that Optuna will minimize.
        """
        # Sample all hyperparameters for this trial
        config = sample_tier1(trial)

        # Log a human-readable summary of this trial's config
        trial_str = (
            f"Trial {trial.number}: "
            f"lr={config['lr']:.1e}, "
            f"mpnn={config['mpnn_num_layers']}L×{config['mpnn_hidden_dim']}d, "
            f"attn={config['attn_num_layers']}L×{config['attn_num_heads']}h, "
            f"token={config['token_dim']}, "
            f"pool={config['pooling_strategy']}, "
            f"lambda={config['lambda_str']}"
        )
        print(f"\n{'─'*70}")
        print(f"  {trial_str}")
        print(f"{'─'*70}")

        # Train the model
        result, _ = train(
            config=config,
            device=device,
            train_path=TRAIN_PATH_STEP1,
            val_path=VAL_PATH,
            seed=SEED,
            trial=trial,
        )

        # Store secondary metrics as "user attributes" on the trial.
        # These aren't used for optimization but are available for post-hoc analysis.
        trial.set_user_attr("best_val_loss_loc", result["best_val_loss_loc"])
        trial.set_user_attr("best_val_loss_str", result["best_val_loss_str"])
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("epochs_trained", result["epochs_trained"])
        trial.set_user_attr("n_params", result["n_params"])
        trial.set_user_attr("runtime_seconds", result["runtime_seconds"])

        # Print trial result
        print(
            f"  -> Metric: {result['best_val_metric']:.6f} "
            f"(loc={result['best_val_loss_loc']:.6f}, str={result['best_val_loss_str']:.6f}) "
            f"@ epoch {result['best_epoch']} | "
            f"Params: {result['n_params']:,} | "
            f"Time: {result['runtime_seconds']:.0f}s"
        )

        # Return the single scalar that Optuna minimizes.
        return result["best_val_metric"]

    # Run the optimization loop. Optuna calls objective() n_trials times.
    study.optimize(objective, n_trials=N_TRIALS_STEP1)

    # Print summary and save top configs for Step 2.
    print_step1_summary(study, results_dir)
    return study


# =============================================================================
# Step 1 Summary
# =============================================================================

def print_step1_summary(study, results_dir):
    """Print Step 1 results and save top configs as JSON for Step 2.

    Also computes lambda calibration: after observing the actual loss magnitudes
    across all completed trials, suggests an updated LAMBDA_METRIC that would give
    the strength task ~20% contribution to the metric. You should check this
    suggestion and update the constant in utils.py if it differs
    significantly from 0.3.
    """
    print("\n" + "=" * 70)
    print("STEP 1 RESULTS")
    print("=" * 70)

    # Filter to only completed trials (not pruned or failed) and sort by metric.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)  # t.value is the return value of objective()
    top_k = min(5, len(completed))  # Take top 5 (or fewer if less completed)

    # Print detailed info for each top config.
    print(f"\nTop {top_k} configurations:")
    print("-" * 70)
    for i, trial in enumerate(completed[:top_k]):
        print(f"\n  #{i+1} — Trial {trial.number} | Metric: {trial.value:.6f}")
        print(f"    val_loss_loc: {trial.user_attrs['best_val_loss_loc']:.6f}")
        print(f"    val_loss_str: {trial.user_attrs['best_val_loss_str']:.6f}")
        print(f"    Params: {trial.user_attrs['n_params']:,}")
        print(f"    Best epoch: {trial.user_attrs['best_epoch']}")
        print(f"    Hyperparameters:")
        for k, v in trial.params.items():
            print(f"      {k}: {v}")

    # Pruning statistics: useful for understanding search efficiency.
    n_complete = len(completed)
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    print(f"\nTrial statistics: {n_complete} completed, {n_pruned} pruned, {n_failed} failed")

    # ---- Lambda Calibration ----
    if completed:
        loc_losses = [t.user_attrs["best_val_loss_loc"] for t in completed]
        str_losses = [t.user_attrs["best_val_loss_str"] for t in completed]
        median_loc = np.median(loc_losses)
        median_str = np.median(str_losses)
        if median_str > 0:
            suggested_lambda = (0.2 / 0.8) * (median_loc / median_str)
            print(f"\n  Lambda calibration:")
            print(f"    Median val_loss_loc: {median_loc:.6f}")
            print(f"    Median val_loss_str: {median_str:.6f}")
            print(f"    Current LAMBDA_METRIC: {LAMBDA_METRIC}")
            print(f"    Suggested LAMBDA_METRIC (for 20% strength contribution): {suggested_lambda:.3f}")
            print(f"    -> Update LAMBDA_METRIC in utils.py if significantly different")

    # ---- Save Top Configs for Step 2 ----
    top_configs = []
    for trial in completed[:top_k]:
        config = dict(trial.params)       # The hyperparameters sampled by suggest_*
        config["trial_number"] = trial.number
        config["metric"] = trial.value
        config["val_loss_loc"] = trial.user_attrs["best_val_loss_loc"]
        config["val_loss_str"] = trial.user_attrs["best_val_loss_str"]
        top_configs.append(config)

    output_path = Path(results_dir) / "step1_top_configs.json"
    with open(output_path, "w") as f:
        json.dump(top_configs, f, indent=2)
    print(f"\nTop configs saved to: {output_path}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_step1()
