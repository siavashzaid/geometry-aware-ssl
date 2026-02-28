from src.training import build_model, evaluate_fn, train_epoch
from src.datasets.precomputed_dataset import precomputedDataset

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from ray import tune

# --- Utility function for ray tune to train model from a config --- #
def ray_train(config):
    # --- Set seeds and device --- #
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    device = config["device"]
    pin_memory = torch.cuda.is_available()

    # --- Load datasets and create dataloader --- #
    train_ds = precomputedDataset(config["train_path"])
    val_ds = precomputedDataset(config["val_path"])

    train_loader = PyGDataLoader(
        train_ds, 
        batch_size=config["train_batch_size"], 
        shuffle=True,  
        num_workers=4, 
        pin_memory=pin_memory
    )

    val_loader = PyGDataLoader(
        val_ds, 
        batch_size=config["val_batch_size"], 
        shuffle=False,  
        num_workers=4, 
        pin_memory=pin_memory
    )

    # --- Initialize model --- #
    model = build_model(config)

    # --- Optimizer and Scheduler --- #
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
    )

    if config['scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config["epochs"], 
            eta_min=config["scheduler_min_lr"]
        )

    # --- Early Stopping Parameters --- #
    best_val_loss = float('inf')
    best_val_loss_loc = float('inf')
    best_val_loss_str = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = config["early_stop_patience"]
    min_delta = config["early_stop_min_delta"]

    # --- Training loop --- #
    for epoch in range(config["epochs"]):
        # --- Training --- #
        train_loss, train_loss_loc, train_loss_str, pred_std = train_epoch(
            model, train_loader, optimizer, device, config["gradient_clip_max_norm"]
        )

        # --- Validation step --- #
        val_loss, val_loss_loc, val_loss_str = evaluate_fn(model, val_loader, device)

        # --- Update best metric and early stopping --- #
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_loss_loc = val_loss_loc
            best_val_loss_str = val_loss_str
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # --- Report metrics to Ray Tune --- #
        tune.report({
            "train_loss": train_loss,
            "train_loss_loc": train_loss_loc,
            "train_loss_str": train_loss_str,
            "val_loss": val_loss,
            "val_loss_loc": val_loss_loc,
            "val_loss_str": val_loss_str,
            "best_val_loss": best_val_loss,
            "best_val_loss_loc": best_val_loss_loc,
            "best_val_loss_str": best_val_loss_str,
            "best_epoch": best_epoch,
            "epoch": epoch,
        })

        # --- Check stopping conditions --- #
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if pred_std < 1e-4:
            print(f"Stopping at epoch {epoch} due to prediction collapse (std={pred_std:.6f})")
            break

        if train_loss < 5e-6:
            print(f"Stopping at epoch {epoch} due to very low training loss ({train_loss:.6f})")
            break

        # --- Step the scheduler --- #
        if config['scheduler']:
            scheduler.step()  #Insert metric if using ReduceLROnPlateau



