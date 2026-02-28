from .utils import ray_train
from ray import tune
import os
import pandas as pd

# --- Top config from Step 1 --- #
base_config = {
      # --- Grid search target --- #
    "lr": tune.grid_search([1e-3, 5e-4, 3e-4, 1e-4, 5e-5 ]),
    # --- Top config Tier 1 params --- #
    "mpnn_hidden_dim":   64,
    "mpnn_num_layers":   4,
    "attn_num_heads":    8,
    "attn_num_layers":   6,
    "token_dim":         64,
    "pooling_strategy":  "mean_pooling",
    # --- Tier 2 (fixed) --- #
    "weight_decay":      0.0,
    "dropout":           0.1,
    "mp_layer_norm":     False,
    # --- Training settings --- #
    "epochs":                  50, #enough to see convergence behavior
    "early_stop_patience":     25, #no early stopping
    "early_stop_min_delta":    1e-5,
    # --- Fixed hyperparameters --- #
    "train_batch_size":        256,
    "val_batch_size":          256,
    "head_mlp_hidden_dim":     256,
    "gradient_clip_max_norm":  1.0,
    "num_output_sources":      1,
    "node_in_dim":             6,
    "edge_in_dim":             6,
    # --- Scheduler (disabled) --- #
    "scheduler":               False,
    "scheduler_min_lr":        5e-6,
    # --- System --- #
    "seed":                    0,
    "device":                  "cuda",
    "train_path":              "/mnt/data/zaid/projects/simulated_data/step_2.h5",
    "val_path":                "/mnt/data/zaid/projects/simulated_data/validation.h5",
}

# --- Tuner --- #
tuner = tune.Tuner(
    tune.with_resources(ray_train, {"gpu": 1, "cpu": 5}),
    param_space=base_config,
    tune_config=tune.TuneConfig(
        metric="best_val_loss",
        mode="min",
        num_samples=1,  # grid search — one run per lr value
    ),
)

if __name__ == "__main__":
    os.makedirs("/mnt/data/zaid/projects/results/lr_search", exist_ok=True)

    results = tuner.fit()

    all_histories = []
    for result in results:
        if result.metrics_dataframe is None:
            continue
        history = result.metrics_dataframe
        trial_id = os.path.basename(result.path)
        history["trial_id"] = trial_id if trial_id else os.path.basename(result.path)
        for k, v in result.config.items():
            history[f"config/{k}"] = v
        all_histories.append(history)

    full_df = pd.concat(all_histories, ignore_index=True)
    full_df.to_csv("/mnt/data/zaid/projects/results/lr_search/lr_search_full_history.csv", index=False)

    # --- Save results --- #
    df = results.get_dataframe()
    df = df.sort_values("best_val_loss", ascending=True)
    df.to_csv("/mnt/data/zaid/projects/results/lr_search/lr_search_results.csv", index=False)

    print("\nLR search complete.")
    print(df[["config/lr", "best_val_loss", "val_loss"]].to_string(index=False))
