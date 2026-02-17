import sys
sys.path.append("../src")
from precomputed_dataset import precomputedDataset
from model import MPNNTransformerModel

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader


# =============================================================================
# Settings  (edit these before running any step)
# =============================================================================

TRAIN_PATH_STEP1  = "path/to/train_10k.h5"
TRAIN_PATH_STEP2  = "path/to/train_25k.h5"
TRAIN_PATH_STEP3A = "path/to/train_100k.h5"
TRAIN_PATH_STEP3B = "path/to/train_500k.h5"

VAL_PATH  = "path/to/val.h5"
TEST_PATH = "path/to/test.h5"       # only used by step3b

RESULTS_DIR = "results/optuna"       # shared directory for inter-step JSON files
DEVICE = None                        # None = auto-detect, or e.g. "cuda:0", "cpu"

N_TRIALS_STEP1 = 80                  # number of Optuna trials in Step 1


# =============================================================================
# Global Constants
# =============================================================================

# LAMBDA_METRIC: Fixed weight for the evaluation metric across ALL trials and steps.
# This is NOT the training loss weight (that's lambda_str, a Tier 1 search param).
# After Step 1's first ~15 trials, the summary will suggest a calibrated value based
# on the observed ratio of val_loss_loc to val_loss_str. Update this constant if the
# suggested value differs significantly from 0.3.
LAMBDA_METRIC = 0.3

SEED = 0

GRADIENT_CLIP_MAX_NORM = 1.0

TIER2_DEFAULTS = {
    "weight_decay": 0.0,
    "dropout": 0.1,
    "mp_layer_norm": False,
}

#Change by hand if needed
HARD_FIXED = {
    "batch_size": 128,
    "head_mlp_hidden_dim": 256,
    "num_output_sources": 1,

    "scheduler": False,
    "scheduler_mode": "min",
    "scheduler_factor": 0.8,
    "scheduler_patience": 20,
    "scheduler_threshold": 1e-5,
    "scheduler_threshold_mode": "rel",
    "scheduler_cooldown": 5,
    "scheduler_min_lr": 1e-6,
}


# =============================================================================
# Device helper
# =============================================================================

def get_device():
    if DEVICE:
        return torch.device(DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Seeding
# =============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Seeds all GPUs (no-op if no CUDA)


# =============================================================================
# Model Construction
# =============================================================================

def build_model(config, sample, device):
    #Construct the MPNN-Transformer model based on the config and sample dimensions.

    model = MPNNTransformerModel(
        node_in_dim=sample.x.shape[-1],
        edge_in_dim=sample.edge_attr.shape[-1],

        num_output_sources=config["num_output_sources"],

        # Tier 1 architecture params (searched in Step 1)
        mpnn_hidden_dim=config["mpnn_hidden_dim"],
        mpnn_num_layers=config["mpnn_num_layers"],
        token_dim=config["token_dim"],
        attn_num_heads=config["attn_num_heads"],
        attn_num_layers=config["attn_num_layers"],
        pooling_strategy=config["pooling_strategy"],

        head_mlp_hidden_dim=config["head_mlp_hidden_dim"],

        # Tier 2 regularization params (fixed at defaults in Step 1/2, searched in Step 3a)
        mp_layer_norm=config["mp_layer_norm"],
        mpnn_dropout=config["mpnn_dropout"],
        attn_dropout=config["attn_dropout"],
        head_dropout=config["head_dropout"],
    ).to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, dataset, device):
    model.eval()
    total_loss_loc = 0.0
    total_loss_str = 0.0
    n_samples = 0

    with torch.no_grad():
        for data in dataset:
            data = data.to(device)

            # Forward pass on a single graph (no batch dimension).
            pred_loc, pred_str = model.forward_from_data(data)
            target_loc = data.y.squeeze(0)
            target_str = data.strength.squeeze(0)

            # Per-sample MSE loss
            total_loss_loc += F.mse_loss(pred_loc, target_loc).item()
            total_loss_str += F.mse_loss(pred_str, target_str).item()
            n_samples += 1

    avg_loss_loc = total_loss_loc / n_samples
    avg_loss_str = total_loss_str / n_samples

    # Combined metric with FIXED lambda (not the trial's training lambda_str)
    avg_metric = avg_loss_loc + LAMBDA_METRIC * avg_loss_str

    return avg_loss_loc, avg_loss_str, avg_metric


# =============================================================================
# Training Loop (single config)
# =============================================================================

def train(config, device, train_path, val_path, seed=0, trial=None):
    set_seed(seed)

    # ---- Data Loading ----
    train_ds = precomputedDataset(train_path)
    val_ds = precomputedDataset(val_path)

    train_loader = PyGDataLoader(
        train_ds,
        batch_size=config["batch_size"], # Fixed
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Model Initialization ----
    sample0 = train_ds[0]
    model = build_model(config, sample0, device)
    n_params = count_parameters(model)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],                   # Tier 1 param: log-uniform [1e-4, 3e-3]
        weight_decay=config["weight_decay"],  # Tier 2 param: {0.0, 1e-4, 1e-3}
        betas=(0.9, 0.999),
    )

    # ---- Learning Rate Scheduler ----
    scheduler = None
    if config["scheduler"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["scheduler_mode"],
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
            threshold=config["scheduler_threshold"],
            threshold_mode=config["scheduler_threshold_mode"],
            cooldown=config["scheduler_cooldown"],
            min_lr=config["scheduler_min_lr"],
        )

    # lambda_str: The TRAINING loss weight for the strength task.
    # NOTE: This is DIFFERENT from LAMBDA_METRIC (the evaluation weight).
    lambda_str = config["lambda_str"]

    # ---- Training History Tracking ----
    history = {
        "train_loss": [],       # Combined training loss: loss_loc + lambda_str * loss_str
        "train_loss_loc": [],
        "train_loss_str": [],
        "val_loss_loc": [],
        "val_loss_str": [],
        "val_metric": [],
        "lr": [],
        "grad_norm": [],
    }

    # ---- Best Model Tracking ----
    best_val_metric = float("inf")
    best_val_loss_loc = float("inf")
    best_val_loss_str = float("inf")
    best_epoch = 0                    # Epoch at which best val_metric was achieved
    best_model_state = None
    patience_counter = 0

    # Step-specific training parameters (set by each run_stepX function):
    max_epochs = config["max_epochs"]
    early_stop_patience = config["early_stop_patience"]

    # Minimum improvement threshold for early stopping.
    min_delta = 1e-5

    start_time = time.time()

    # ==================== EPOCH LOOP ====================
    for epoch in range(1, max_epochs + 1):

        # ---- Training Phase ----
        model.train()

        # Accumulators for epoch-level metrics (reset each epoch)
        epoch_loss = 0.0
        epoch_loss_loc = 0.0
        epoch_loss_str = 0.0
        epoch_preds = []

        # ---- Batch Loop ----
        for data in train_loader:
            # Move the PyG Batch object to GPU. This includes:
            data = data.to(device)

            # Forward pass through the full pipeline:
            pred_loc, pred_str = model.forward_from_data(data)

            # Multi-task loss: location MSE + lambda_str * strength MSE
            loss_loc = F.mse_loss(pred_loc, data.y)
            loss_str = F.mse_loss(pred_str, data.strength)
            loss = loss_loc + lambda_str * loss_str

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)

            optimizer.step()  # Update weights using AdamW

            # Accumulate batch losses
            epoch_loss += loss.item()
            epoch_loss_loc += loss_loc.item()
            epoch_loss_str += loss_str.item()

            # Save predictions for collapse detection
            epoch_preds.append(pred_loc.detach().cpu().numpy())

        # ---- Epoch-Level Metrics ----
        n_batches = len(train_loader)
        avg_train_loss = epoch_loss / n_batches
        avg_train_loc = epoch_loss_loc / n_batches
        avg_train_str = epoch_loss_str / n_batches

        # ---- Gradient Norm ----
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5

        # ---- Prediction Collapse Detection ----
        pred_std = np.concatenate(epoch_preds, axis=0).std(axis=0).mean()
        if pred_std < 0.004:
            print(f"    [!] Predictions collapsed at epoch {epoch}")
            break  # Abort this trial -- the model is useless

        # ---- Validation ----
        # Evaluate on the validation set using the FIXED LAMBDA_METRIC.
        # This metric is what Optuna optimizes and what drives early stopping.
        val_loss_loc, val_loss_str, val_metric = evaluate(model, val_ds, device)

        # ---- Record History ----
        history["train_loss"].append(avg_train_loss)
        history["train_loss_loc"].append(avg_train_loc)
        history["train_loss_str"].append(avg_train_str)
        history["val_loss_loc"].append(val_loss_loc)
        history["val_loss_str"].append(val_loss_str)
        history["val_metric"].append(val_metric)
        history["lr"].append(optimizer.param_groups[0]["lr"])  # Current LR (may change via scheduler)
        history["grad_norm"].append(grad_norm)

        # ---- Scheduler Step ----
        if scheduler is not None:
            scheduler.step(val_metric)

        # ---- Best Model Tracking ----
        if val_metric < best_val_metric - min_delta:
            best_val_metric = val_metric
            best_val_loss_loc = val_loss_loc
            best_val_loss_str = val_loss_str
            best_epoch = epoch
            patience_counter = 0  # Reset patience -- we just improved
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1  # No improvement -- increment patience

        # ---- Logging ----
        if epoch == 1 or epoch % 10 == 0 or patience_counter >= early_stop_patience:
            print(
                f"    Epoch {epoch:4d} | "
                f"Train: {avg_train_loss:.6f} (loc={avg_train_loc:.6f}, str={avg_train_str:.6f}) | "
                f"Val metric: {val_metric:.6f} (loc={val_loss_loc:.6f}, str={val_loss_str:.6f}) | "
                f"Best: {best_val_metric:.6f} @ ep{best_epoch} | "
                f"Patience: {patience_counter}/{early_stop_patience} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        # ---- Optuna Pruning (Step 1 only) ----
        if trial is not None:
            trial.report(val_metric, epoch)
            if trial.should_prune():
                print(f"    [Pruned] Trial pruned at epoch {epoch}")
                import optuna
                raise optuna.TrialPruned()

        # ---- Early Stopping ----
        if patience_counter >= early_stop_patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    # ==================== END EPOCH LOOP ====================

    runtime = time.time() - start_time

    # ---- Assemble Results ----
    result = {
        "best_val_metric": best_val_metric,
        "best_val_loss_loc": best_val_loss_loc,
        "best_val_loss_str": best_val_loss_str,
        "best_epoch": best_epoch,
        "epochs_trained": len(history["train_loss"]),
        "n_params": n_params,
        "runtime_seconds": runtime,
        "history": history,
        "config": config.copy(),
    }

    # ---- GPU Memory Cleanup ----
    del model, optimizer, train_loader, train_ds, val_ds
    torch.cuda.empty_cache()

    return result, best_model_state
