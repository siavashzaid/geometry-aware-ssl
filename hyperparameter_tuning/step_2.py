from utils import run_training
import pandas as pd
import os

#TODO IMPLEMENT LR EITHER BASE OR TOP CONFIGS

# --- Base config (shared params across all trials) --- #
base_config = {
    # --- Tier 2 regularization (fixed in step 2) --- #
    "weight_decay": 0.0,
    "dropout": 0.1,
    "mp_layer_norm": False,
    # --- Step-specific settings --- #
    "epochs": 120,
    "train_path": "/mnt/data/zaid/projects/simulated_data/step2.h5",
    "val_path": "/mnt/data/zaid/projects/simulated_data/validation.h5",
    "early_stop_patience": 40,
    "early_stop_min_delta": 1e-5,
    # --- Fixed hyperparameters --- #
    "train_batch_size": 128,
    "val_batch_size": 256,
    "head_mlp_hidden_dim": 256,
    "gradient_clip_max_norm": 1.0,
    # --- Scheduler (disabled) --- #
    "scheduler": False,
    "scheduler_min_lr": 5e-6,
    # --- System --- #
    "seed": 0,
    "device": "cuda",
    # --- Model Settings --- #
    "num_output_sources": 1,
    "node_in_dim": 6,
    "edge_in_dim": 6,
}

# --- Top 5 configs from Step 1 (Tier 1 params only) --- #
tier1_configs = [ 
    {"lr": 9, "mpnn_num_layers": 4, "attn_num_layers": 6, "attn_num_heads": 8, "token_dim": 64, "mpnn_hidden_dim": 64, "pooling_strategy": "mean_pooling"},

    {"lr": 9, "mpnn_num_layers": 4, "attn_num_layers": 6, "attn_num_heads": 8, "token_dim": 128, "mpnn_hidden_dim": 64, "pooling_strategy": "mean_pooling"},

    {"lr": 9, "mpnn_num_layers": 4, "attn_num_layers": 4, "attn_num_heads": 4, "token_dim": 128, "mpnn_hidden_dim": 256, "pooling_strategy": "mean_pooling"},

    {"lr": 9, "mpnn_num_layers": 4, "attn_num_layers": 2, "attn_num_heads": 4, "token_dim": 128, "mpnn_hidden_dim": 256, "pooling_strategy": "cls_token"},

    {"lr": 9, "mpnn_num_layers": 4, "attn_num_layers": 2, "attn_num_heads": 4, "token_dim": 64, "mpnn_hidden_dim": 64, "pooling_strategy": "mean_pooling"},
]

# --- Merge base config with tier 1 configs --- #
top_configs = [{**base_config, **tier1} for tier1 in tier1_configs]


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    all_histories = []
    all_final     = []

    for i, config in enumerate(top_configs):
        print(f"\n{'='*60}")
        print(f"Step 2 | Trial {i+1}/{5}")
        print(f"{'='*60}")

        history_df, final_metrics = run_training(config)

        # --- Attach metadata --- #
        history_df["step1_rank"] = i + 1
        for k, v in config.items():
            history_df[f"config/{k}"] = v
        all_histories.append(history_df)

        final_metrics["step1_rank"] = i + 1
        for k, v in config.items():
            final_metrics[f"config/{k}"] = v
        all_final.append(final_metrics)

    # --- Save full history --- #
    full_df = pd.concat(all_histories, ignore_index=True)
    full_df.to_csv("results/step2_full_history.csv", index=False)

    # --- Save final results with step 2 ranking --- #
    final_df = pd.DataFrame(all_final)
    final_df = final_df.sort_values("best_val_loss", ascending=True).reset_index(drop=True)
    final_df["step2_rank"] = range(1, len(final_df) + 1)
    final_df.to_csv("results/step2_results.csv", index=False)

    print("\nStep 2 complete. Results saved to results/step2_results.csv")
    print(final_df[["step1_rank", "step2_rank", "best_val_loss"]].to_string(index=False))
