from utils import build_model, evaluate_fn, train_fn

from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

import os
import pandas as pd

# --- Search parameters --- #
search_space = {
    # --- Tier 1 params --- #
    "lr": tune.loguniform(1e-4, 1e-3),
    'mpnn_hidden_dim': tune.choice([64, 128, 256]),
    'mpnn_num_layers': tune.choice([1, 2, 3, 4]),
    'attn_num_heads': tune.choice([2, 4, 8]), #embed_dim must be divisible by num_heads, so if we stay at 128 token dim, 2n
    'attn_num_layers': tune.choice([2, 4, 6, 8]),
    'token_dim': tune.choice([64, 128, 256]),
    'pooling_strategy': tune.choice(['cls_token', 'mean_pooling']),

    # --- Tier 2 regularization params --- #
    "weight_decay": 0.0,
    "dropout": 0.1,
    "mp_layer_norm": False,
    
    # --- FIXED PARAMS (not searched in any step) ---
    # --- Tuning Settings --- #
    "epochs": 2, #TODO: CHANGE THIS
    # --- Early Stopping Settings --- #
    "early_stop_patience": 25,
    "early_stop_min_delta": 1e-5,
    # --- Fixed Hyperparameters --- #
    "train_batch_size": 128,
    "val_batch_size": 256,
    "head_mlp_hidden_dim": 256,
    "gradient_clip_max_norm": 1.0,
    # --- Scheduler Settings --- #
    "scheduler": False,
    "scheduler_mode": "min",
    "scheduler_factor": 0.8,
    "scheduler_patience": 20,
    "scheduler_threshold": 1e-5,
    "scheduler_threshold_mode": "rel",
    "scheduler_cooldown": 5,
    "scheduler_min_lr": 1e-6,
    # --- Data and System Settings --- #
    "seed": 0,
    "device": "cuda",
    "train_path": "/mnt/data/zaid/projects/simulated_data/step1.h5",
    "val_path": "/mnt/data/zaid/projects/simulated_data/validation.h5",
    # --- Model Settings --- #
    "num_output_sources": 1,
    "node_in_dim": 6,
    "edge_in_dim": 6,
}

# --- TPE Search Algorithm --- #
search_alg = HyperOptSearch(
    metric="val_loss",
    mode="min",
)

# --- Tuner Setup --- #
tuner = tune.Tuner(
        tune.with_resources(train_fn, {"gpu": 1, "cpu": 5}), #4 worker threads + 1 main thread
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=3, # 80
            search_alg=search_alg,
            scheduler=ASHAScheduler(
                max_t=2, #120
                grace_period=1, #20  
                reduction_factor=2,  
            ),  
        ),
    )

if __name__ == "__main__":

    # --- Run Hyperparameter Tuning --- #
    results = tuner.fit()

    # --- save full training history for all trials to a CSV --- #
    all_histories = []
    for result in results:
        history = result.metrics_dataframe
        history["trial_id"] = os.path.basename(result.path)
        # add config values
        for k, v in result.config.items():
            history[f"config/{k}"] = v
        all_histories.append(history)

    full_df = pd.concat(all_histories, ignore_index=True)
    full_df.to_csv("results/tune_full_history.csv", index=False)

    # --- save last results --- #
    df = results.get_dataframe()
    df.to_csv("results/tune_results.csv", index=False)