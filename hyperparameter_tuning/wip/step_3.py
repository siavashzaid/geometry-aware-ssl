from utils import train_fn
import os
import pandas as pd
from ray import tune
import itertools

# --- Settings --- #
STEP2_RESULTS_PATH = "results/step2_results.csv"
TOP_N = 1  # Use best config from step 2

# --- Tier 2 Grid --- #
tier2_grid = {
    "weight_decay": [0.0, 1e-4, 1e-3],
    "dropout":      [0.0, 0.05, 0.1],
    "mp_layer_norm": [True, False],
}

# --- Load best config from Step 2 --- #
step2_df = pd.read_csv(STEP2_RESULTS_PATH)
step2_df = step2_df.sort_values("val_loss", ascending=True).head(TOP_N)                             #TODO:Maybe change to best_val_loss

config_cols = [col for col in step2_df.columns if col.startswith("config/")]
base_config = {col.replace("config/", ""): step2_df.iloc[0][col] for col in config_cols}

# --- Override step-specific settings --- #
base_config["epochs"] = 2                                                                           # TODO: CHANGE THIS 
base_config["train_path"] = "/mnt/data/zaid/projects/simulated_data/step3a.h5"  
base_config["val_path"] = "/mnt/data/zaid/projects/simulated_data/validation.h5"  
base_config["early_stop_patience"] = 25

# --- Build all 18 grid combinations --- #
tier2_keys = list(tier2_grid.keys())
tier2_values = list(tier2_grid.values())
grid_combinations = [
    dict(zip(tier2_keys, combo))
    for combo in itertools.product(*tier2_values)
]  

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    all_histories = []
    all_final = []

    for i, tier2_params in enumerate(grid_combinations):
        print(f"\n--- Running Step 3a combination {i+1}/{len(grid_combinations)}: {tier2_params} ---")

        # --- Merge locked Tier 1 config with current Tier 2 combination --- #
        config = {**base_config, **tier2_params}

        tuner = tune.Tuner(
            tune.with_resources(train_fn, {"gpu": 1, "cpu": 5}),
            param_space=config,  
            tune_config=tune.TuneConfig(
                metric="val_loss",
                mode="min",
                num_samples=1,  
                
            ),
        )

        results = tuner.fit()

        for result in results:
            if result.metrics_dataframe is None:  
                continue
            history = result.metrics_dataframe
            history["trial_id"] = os.path.basename(result.path)
            history["grid_combination"] = i + 1
            for k, v in result.config.items():
                history[f"config/{k}"] = v
            all_histories.append(history)

        # --- Save final metrics for ranking --- #
        summary = results.get_dataframe()
        summary["grid_combination"] = i + 1
        # --- Add tier 2 params as separate columns for easy inspection --- #
        for k, v in tier2_params.items():
            summary[f"tier2/{k}"] = v
        all_final.append(summary)

    # --- Save full training history --- #
    full_df = pd.concat(all_histories, ignore_index=True)
    full_df.to_csv("results/step3a_full_history.csv", index=False)

    # --- Save final results with ranking --- #
    final_df = pd.concat(all_final, ignore_index=True)
    final_df = final_df.sort_values("val_loss", ascending=True)
    final_df["step3a_rank"] = range(1, len(final_df) + 1)
    final_df.to_csv("results/step3a_results.csv", index=False)

    print("\nStep 3a complete. Results saved to results/step3a_results.csv")
    print(final_df[["grid_combination", "step3a_rank", "val_loss"] +
                   [f"tier2/{k}" for k in tier2_keys]].to_string(index=False))
