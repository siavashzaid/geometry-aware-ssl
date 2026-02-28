from .utils import ray_train
from ray import tune
import os
import pandas as pd

# --- Search space --- #
search_space = {
    # --- Regularization grid search --- # TODO Set this
    "lr":            tune.grid_search([5e-4, 3e-4, 1e-4]),
    "weight_decay":  tune.grid_search([0.0, 1e-4, 1e-3]),
    "dropout":       tune.grid_search([0.0, 0.1, 0.2]), 
    "mp_layer_norm": tune.grid_search([True, False]),
    # --- Best Tier 1 settings --- #
    "mpnn_hidden_dim":  64,
    "mpnn_num_layers":  4,
    "attn_num_heads":   8,
    "attn_num_layers":  6,
    "token_dim":        64,
    "pooling_strategy": "mean_pooling",
    # --- Training settings --- #
    "epochs":                  120,
    "early_stop_patience":     40,
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
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="best_val_loss",
        mode="min",
        num_samples=1,
    ),
)

if __name__ == "__main__":

    os.makedirs("/mnt/data/zaid/projects/results/step_2", exist_ok=True)

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
    full_df.to_csv("/mnt/data/zaid/projects/results/step_2/full_history.csv", index=False)
    
    df = results.get_dataframe()
    df = df.sort_values("best_val_loss", ascending=True)
    df.to_csv("/mnt/data/zaid/projects/results/step_2/results.csv", index=False)

    print("\nStep 2 complete. Results saved to step2_results.csv")
    print(df[["config/weight_decay", "config/dropout", "config/mp_layer_norm", "best_val_loss"]].to_string(index=False))

