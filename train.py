from src.training import run_training
from configs.final_config import final_config

import os

OUTPUT_DIR = '/mnt/data/zaid/projects/results/training'
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'best_model.pt')

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    history_df, final_metrics = run_training(final_config, checkpoint_path=CHECKPOINT_PATH)

    # --- Save results --- #
    history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

    # --- Print final metrics --- #
    print("Final Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.6f}")
