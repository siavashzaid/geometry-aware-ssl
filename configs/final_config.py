# --- Final configuration for training the geometry-aware SSL model --- #
final_config = {
    # --- Data  --- #
    "num_output_sources":      1,
    "node_in_dim":             6,
    "edge_in_dim":             6,
    "train_path":              "/mnt/data/zaid/projects/simulated_data/single_geometry_train.h5",
    "val_path":                "/mnt/data/zaid/projects/simulated_data/single_geometry_val.h5",
    
    # --- Architecture --- #
    "mpnn_hidden_dim":  64,
    "mpnn_num_layers":  4,
    "token_dim":        64,
    "attn_num_heads":   8,
    "attn_num_layers":  6,
    "head_mlp_hidden_dim":     256,
    "pooling_strategy": "mean_pooling",
    
    # --- Regularization parameters --- 
    "weight_decay":  0.0001,
    "dropout":       0.1, 
    "mp_layer_norm": False,

    # --- Optimizer Settings --- #
    "lr":            5e-4,
    "scheduler":               False,
    "scheduler_min_lr":        5e-6,
    "gradient_clip_max_norm":  1.0,
 
    # --- Training settings --- #
    "epochs":                  200,
    "early_stop_patience":     50,
    "early_stop_min_delta":    1e-5,
    "train_batch_size":        256,
    "val_batch_size":          256,

    # --- System --- #
    "seed":                    0,
    "device":                  "cuda:2",
}