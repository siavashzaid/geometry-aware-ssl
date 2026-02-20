from utils import build_model, evaluate_fn, train_fn

from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

import os
import pandas as pd

from utils import train_fn
import os
import pandas as pd
from ray import tune, train
from ray.tune.search import BasicVariantGenerator

# --- Settings --- #
TOP_N = 5
STEP1_RESULTS_PATH = "results/tune_results.csv"

# --- Load top N configs from Step 1 --- #
step1_df = pd.read_csv(STEP1_RESULTS_PATH)
step1_df = step1_df.sort_values("val_loss", ascending=True).head(TOP_N)                         #TODO:Maybe change to best_val_loss

# --- Extract config columns from Step 1 results --- #
config_cols = [col for col in step1_df.columns if col.startswith("config/")]
top_configs = []
for _, row in step1_df.iterrows():
    config = {col.replace("config/", ""): row[col] for col in config_cols}
    # --- Override step-specific settings --- #
    config["epochs"] = 2                                                                        # TODO: CHANGE THIS (25k run, same max_t as step 1)
    config["train_path"] = "/mnt/data/zaid/projects/simulated_data/step2.h5"  
    config["val_path"] = "/mnt/data/zaid/projects/simulated_data/validation.h5"  
    config["early_stop_patience"] = 25
    top_configs.append(config)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    all_histories = []
    all_final = []

    for i, config in enumerate(top_configs):
        print(f"\n--- Running Step 2 trial {i+1}/{TOP_N} ---")

        tuner = tune.Tuner(
            tune.with_resources(train_fn, {"gpu": 1, "cpu": 5}),
            param_space=config,  # fixed config, no search
            tune_config=tune.TuneConfig(
                metric="val_loss",
                mode="min",
                num_samples=1,  
            ),
        )

        results = tuner.fit()

        for result in results:
            if result.metrics_dataframe is None:  # skip if trial crashed before reporting
                continue
            history = result.metrics_dataframe
            history["trial_id"] = os.path.basename(result.path)
            history["step1_rank"] = i + 1  # rank from step 1 (1 = best)
            for k, v in result.config.items():
                history[f"config/{k}"] = v
            all_histories.append(history)

        # --- Save final metrics for ranking comparison --- #
        summary = results.get_dataframe()
        summary["step1_rank"] = i + 1
        all_final.append(summary)

    # --- Save full training history --- #
    full_df = pd.concat(all_histories, ignore_index=True)
    full_df.to_csv("results/step2_full_history.csv", index=False)

    # --- Save final results with step 2 ranking --- #
    final_df = pd.concat(all_final, ignore_index=True)
    final_df = final_df.sort_values("val_loss", ascending=True)
    final_df["step2_rank"] = range(1, len(final_df) + 1)
    final_df.to_csv("results/step2_results.csv", index=False)

    print("\nStep 2 complete. Results saved to results/step2_results.csv")
    print(final_df[["step1_rank", "step2_rank", "val_loss"]].to_string(index=False))
