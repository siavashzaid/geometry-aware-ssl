import sys
sys.path.append("../src")
from precomputed_dataset import precomputedDataset
from modules import MPNNLayer, MPNNTokenizer, SelfAttentionEncoder, PredictionHead
from model import MPNNTransformerModel

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader

from ray import tune
from ray.tune.schedulers import ASHAScheduler

# --- Utility function to build model from config---
def build_model(config):
    model = MPNNTransformerModel(
        # Fixed architecture params (not searched in any step)
        node_in_dim=config["node_in_dim"],
        edge_in_dim=config["edge_in_dim"],
        num_output_sources=config["num_output_sources"],
        # Tier 1 architecture params (searched in Step 1)
        mpnn_hidden_dim=config["mpnn_hidden_dim"],
        mpnn_num_layers=config["mpnn_num_layers"],
        token_dim=config["token_dim"],
        attn_num_heads=config["attn_num_heads"],
        attn_num_layers=config["attn_num_layers"],
        pooling_strategy=config["pooling_strategy"],
        head_mlp_hidden_dim=config["head_mlp_hidden_dim"],
        # Tier 2 regularization params (fixed at defaults in Step 1 and 2, searched in Step 3a)
        mp_layer_norm=config["mp_layer_norm"],
        mpnn_dropout=config["dropout"],
        attn_dropout=config["dropout"],
        head_dropout=config["dropout"],
    ).to(config["device"])
    return model

# --- Utility function to evaluate model on validation set ---
def evaluate_fn(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_loss_loc = 0.0
    total_loss_str = 0.0
    num_samples = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred_loc, pred_str = model.forward_from_data(data) 

            total_loss_loc += F.mse_loss(pred_loc, data.y, reduction='sum').item()
            total_loss_str += F.mse_loss(pred_str, data.strength, reduction='sum').item()

            batch_size = pred_loc.size(0)
            num_samples += batch_size
         
    total_loss = total_loss_loc + total_loss_str

    val_loss = total_loss / num_samples
    val_loss_loc = total_loss_loc / num_samples
    val_loss_str = total_loss_str / num_samples

    return val_loss, val_loss_loc, val_loss_str

# --- Utility function to train model from a config ---
def train_fn(config):
    # --- Debug print statement ---
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}\n")

    # --- Set seeds and device ---
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    device = config["device"]

    # --- Load datasets and create dataloader ---
    train_ds = precomputedDataset(config["train_path"])
    val_ds = precomputedDataset(config["val_path"])

    train_loader = PyGDataLoader(
        train_ds, 
        batch_size=config["train_batch_size"], 
        shuffle=True,  
        num_workers=4, 
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = PyGDataLoader(
        val_ds, 
        batch_size=config["val_batch_size"], 
        shuffle=False,  
        num_workers=4, 
        pin_memory=True if torch.cuda.is_available() else False
    )

    # --- Initialize model ---
    model = build_model(config)

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
    )

    if config['scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['scheduler_mode'],
            factor=config['scheduler_factor'],          
            patience=config['scheduler_patience'],       
            threshold=config['scheduler_threshold'],    
            threshold_mode=config['scheduler_threshold_mode'],
            cooldown=config['scheduler_cooldown'],        
            min_lr=config['scheduler_min_lr'],
        )

    # --- Early Stopping Parameters ---
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config["early_stop_patience"]
    min_delta = config["early_stop_min_delta"]

    # --- Training loop ---
    for epoch in range(config["epochs"]):

        model.train()
        train_loss = 0.0
        train_loss_loc = 0.0
        train_loss_str = 0.0

        # --- for monitoring prediction collapse ---
        pred_locs = []

        for data in train_loader:
            data = data.to(device)
            pred_loc, pred_str = model.forward_from_data(data) 

            loss_loc = F.mse_loss(pred_loc, data.y)
            loss_str = F.mse_loss(pred_str, data.strength)
            loss = loss_loc + loss_str

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_max_norm"])
            optimizer.step()

            train_loss += loss.item()
            train_loss_loc += loss_loc.item()
            train_loss_str += loss_str.item()

            pred_locs.append(pred_loc.detach().cpu())

        # --- compute epoch-level metrics ---
        train_loss /= len(train_loader)
        train_loss_loc /= len(train_loader)
        train_loss_str /= len(train_loader)

        pred_std = torch.cat(pred_locs, dim=0).std(dim=0).mean().item()

        # --- Validation step ---
        val_loss, val_loss_loc, val_loss_str = evaluate_fn(model, val_loader, device)

        # --- Update best metric and early stopping ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_loss_loc = val_loss_loc
            best_val_loss_str = val_loss_str
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # --- Report metrics to Ray Tune ---
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

        # --- Check stopping conditions ---
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if pred_std < 1e-4:
            print(f"Stopping at epoch {epoch} due to prediction collapse (std={pred_std:.6f})")
            break

        if train_loss < 5e-6:
            print(f"Stopping at epoch {epoch} due to very low training loss ({train_loss:.6f})")
            break

        # --- Step the scheduler ---
        if config['scheduler']:
            scheduler.step(val_loss) 