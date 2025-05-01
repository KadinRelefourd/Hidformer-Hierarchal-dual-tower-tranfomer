import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import os  # Import os


# from data_pipeline import DataPipeline
from preprocessing import createDataset
from hidformer import Hidformer


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=5,
        verbose=False,
        delta=0,
        path="./model/checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: './model/checkpoint.pt'
            trace_func (function): trace print function.
                                   Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # Ensure model directory exists
        model_dir = os.path.dirname(self.path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory for model checkpoint: {model_dir}")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def main():
    # --- Configuration ---
    TICKER_LIST_CSV = "./src/tickers.csv"  # Path to your ticker list
    DATA_ROOT_DIR = "./data/"  # Root directory for raw/processed data
    MODEL_SAVE_PATH = (
        "./model/test_larger_dataset_128.pt"  # Path to save best model (updated name)
    )
    # SCALER_SAVE_PATH = "./model/input_scaler.joblib"  # Path to save the fitted scaler

    # Data parameters
    LOOKBACK_WINDOW = 128  # Lookback window (Hidformer input length T_in)
    PREDICTION_HORIZON = 128  # Forecast horizon (Hidformer pred_len H)
    # Features to use (ensure these match columns in downloaded CSVs from getData)
    FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
    TARGET_COLUMN = "Close"  # Target column for prediction (can be a list too)

    # Model parameters (match Hidformer definition)
    TOKEN_LENGTH = 32
    STRIDE = 16
    NUM_TIME_BLOCKS = 4
    NUM_FREQ_BLOCKS = 2
    D_MODEL = 128
    FREQ_K = 64
    DROPOUT = 0.2
    MERGE_MODE = "linear"
    MERGE_K = 2

    # Training parameters
    BATCH_SIZE = 64  # Adjusted from previous train.py context
    LEARNING_RATE = 1e-4
    EPOCHS = 50  # Max epochs, early stopping will likely trigger sooner
    EARLY_STOPPING_PATIENCE = 5  # From Hidformer paper

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Preparation using preprocessing.py ---
    print("--- Preparing Data ---")
    # Read tickers from CSV
    try:
        print("Loading/Creating train dataset (unscaled)...")
        train_ds = createDataset(  # <<< Use train_ds directly
            ticker_list_csv_path=TICKER_LIST_CSV,
            root=DATA_ROOT_DIR,
            lookback_window=LOOKBACK_WINDOW,
            prediction_horizon=PREDICTION_HORIZON,
            split="train",
            target_column=TARGET_COLUMN,
            feature_columns=FEATURE_COLUMNS,
            download=True,
            force_regenerate=False,
        )
        print("Loading/Creating validation dataset (unscaled)...")
        val_ds = createDataset(  # <<< Use val_ds directly
            ticker_list_csv_path=TICKER_LIST_CSV,
            root=DATA_ROOT_DIR,
            lookback_window=LOOKBACK_WINDOW,
            prediction_horizon=PREDICTION_HORIZON,
            split="val",
            target_column=TARGET_COLUMN,
            feature_columns=FEATURE_COLUMNS,
            download=False,
            force_regenerate=False,
        )
        print("Loading/Creating test dataset (unscaled)...")
        test_ds = createDataset(  # <<< Use test_ds directly
            ticker_list_csv_path=TICKER_LIST_CSV,
            root=DATA_ROOT_DIR,
            lookback_window=LOOKBACK_WINDOW,
            prediction_horizon=PREDICTION_HORIZON,
            split="test",
            target_column=TARGET_COLUMN,
            feature_columns=FEATURE_COLUMNS,
            download=False,
            force_regenerate=False,
        )
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}"
    )
    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    # --- Model Instantiation ---
    # Determine input_dim based on selected features
    input_dim = len(FEATURE_COLUMNS)
    print(f"Model input dimension (features): {input_dim}")

    model = Hidformer(
        input_dim=input_dim,
        pred_len=PREDICTION_HORIZON,
        token_length=TOKEN_LENGTH,
        stride=STRIDE,
        num_time_blocks=NUM_TIME_BLOCKS,
        num_freq_blocks=NUM_FREQ_BLOCKS,
        d_model=D_MODEL,
        freq_k=FREQ_K,
        dropout=DROPOUT,
        merge_mode=MERGE_MODE,
        merge_k=MERGE_K,
    ).to(device)
    print(
        f"Model instantiated with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    # --- Loss and Optimizer ---
    # criterion = nn.MSELoss()  # Use MSE as per Hidformer paper
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- <<< ADD WEIGHTED MSE SETUP >>> ---
    # Create weights: tensor from H down to 1
    weights = torch.linspace(PREDICTION_HORIZON, 1, PREDICTION_HORIZON)
    # Normalize weights so the average weight is 1.0
    # This keeps the overall scale of the loss similar to standard MSE
    weights = weights / torch.mean(weights)
    # Reshape for broadcasting: (1, H, 1) -> matches (B, H, C) tensors
    # and move to the correct device
    weights = weights.view(1, -1, 1).to(device)
    print(
        f"Using weighted MSE with weights: {weights.squeeze().cpu().numpy()}"
    )  # Print weights for verification
    # --- <<< END WEI

    # --- Early Stopping Initialization ---
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, verbose=True, path=MODEL_SAVE_PATH
    )

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        batch_count = 0
        for X, y in train_loader:  # X is now scaled, y is original
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # Model forward pass
            # Input X is scaled
            # Output pred is denormalized by RevIN back to original scale
            pred = model(X)

            # --- Loss Calculation ---
            # Compare denormalized prediction 'pred' with original target 'y'
            if y.shape[-1] != pred.shape[-1]:
                # This logic is needed if y only contains the target column,
                # but model predicts all features.
                if isinstance(TARGET_COLUMN, str):
                    try:
                        target_indices = [FEATURE_COLUMNS.index(TARGET_COLUMN)]
                    except ValueError:
                        raise ValueError(
                            f"TARGET_COLUMN '{TARGET_COLUMN}' not found in FEATURE_COLUMNS {FEATURE_COLUMNS}"
                        )
                else:  # List of targets
                    target_indices = [FEATURE_COLUMNS.index(tc) for tc in TARGET_COLUMN]

                if len(target_indices) != y.shape[-1]:
                    raise ValueError(
                        f"Shape mismatch: Target y has {y.shape[-1]} features, but found {len(target_indices)} target indices."
                    )
                # Select the corresponding column(s) from the denormalized prediction
                pred_for_loss = pred[:, :, target_indices]
            else:
                # Assume y contains all features (unscaled), matching prediction (denormalized)
                pred_for_loss = pred

            # Check shapes before loss
            if pred_for_loss.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch before loss: pred={pred_for_loss.shape}, target={y.shape}"
                )

            # Loss is calculated between denormalized prediction and original target
            # --- Weighted MSE Calculation ---
            # Calculate element-wise squared error
            squared_errors = (pred_for_loss - y) ** 2
            # Apply weights (broadcasting along Batch and Channel dimensions)
            weighted_squared_errors = squared_errors * weights
            # Calculate the mean over all elements
            loss = torch.mean(weighted_squared_errors)
            # --- End Weighted MSE Calculation ---

            # loss = criterion(pred_for_loss, y) # REMOVE or COMMENT OUT old line

            # loss = criterion(pred_for_loss, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1

        avg_train_loss = train_loss / batch_count

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for Xv, yv in val_loader:  # Xv is scaled, yv is original
                Xv, yv = Xv.to(device), yv.to(device)
                pv = model(Xv)  # pv is denormalized prediction

                # Select target features from denormalized prediction if necessary
                if yv.shape[-1] != pv.shape[-1]:
                    if isinstance(TARGET_COLUMN, str):
                        target_indices = [FEATURE_COLUMNS.index(TARGET_COLUMN)]
                    else:  # List of targets
                        target_indices = [
                            FEATURE_COLUMNS.index(tc) for tc in TARGET_COLUMN
                        ]
                    pv_for_loss = pv[:, :, target_indices]
                else:
                    pv_for_loss = pv

                if pv_for_loss.shape != yv.shape:
                    raise ValueError(
                        f"Shape mismatch during validation: pred={pv_for_loss.shape}, target={yv.shape}"
                    )

                # --- Weighted MSE Calculation ---
                val_squared_errors = (pv_for_loss - yv) ** 2
                val_weighted_squared_errors = val_squared_errors * weights
                val_batch_loss = torch.mean(val_weighted_squared_errors)
                # --- End Weighted MSE Calculation ---

                # val_loss += criterion(pv_for_loss, yv).item() # REMOVE or COMMENT OUT old line
                val_loss += val_batch_loss.item()  # Add the calculated weighted loss
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        print(
            f"Epoch {epoch:02d}/{EPOCHS} — Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # --- Early Stopping Check ---
        # Note: Early stopping is still based on the loss in the original price scale
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\n--- Training Finished ---")

    # --- Testing Loop ---
    print("\n--- Starting Testing ---")
    # Load the best model saved by early stopping
    try:
        # Ensure model directory exists before loading
        model_dir = os.path.dirname(MODEL_SAVE_PATH)
        if model_dir and not os.path.exists(model_dir):
            print(
                f"Warning: Model save directory {model_dir} not found during testing load."
            )
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded best model from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(
            f"Could not load best model from {MODEL_SAVE_PATH}: {e}. Testing with last model state."
        )

    model.eval()
    test_loss = 0.0
    test_batch_count = 0
    with torch.no_grad():
        for Xt, yt in test_loader:  # Xt is scaled, yt is original
            Xt, yt = Xt.to(device), yt.to(device)
            pt = model(Xt)  # pt is denormalized prediction

            # Select target features from denormalized prediction if necessary
            if yt.shape[-1] != pt.shape[-1]:
                if isinstance(TARGET_COLUMN, str):
                    target_indices = [FEATURE_COLUMNS.index(TARGET_COLUMN)]
                else:  # List of targets
                    target_indices = [FEATURE_COLUMNS.index(tc) for tc in TARGET_COLUMN]
                pt_for_loss = pt[:, :, target_indices]
            else:
                pt_for_loss = pt

            if pt_for_loss.shape != yt.shape:
                raise ValueError(
                    f"Shape mismatch during testing: pred={pt_for_loss.shape}, target={yt.shape}"
                )

            # Loss calculated between denormalized prediction and original target
            # --- Weighted MSE Calculation ---
            test_squared_errors = (pt_for_loss - yt) ** 2
            test_weighted_squared_errors = test_squared_errors * weights
            test_batch_loss = torch.mean(test_weighted_squared_errors)
            # --- End Weighted MSE Calculation ---
            test_loss += test_batch_loss.item()
            test_batch_count += 1

    avg_test_loss = test_loss / test_batch_count
    print(f"Test MSE: {avg_test_loss:.6f}")

    # --- Optional: Add MAE calculation for testing ---
    test_mae = 0.0
    with torch.no_grad():
        for Xt, yt in test_loader:  # Xt is scaled, yt is original
            Xt, yt = Xt.to(device), yt.to(device)
            pt = model(Xt)  # pt is denormalized prediction
            if yt.shape[-1] != pt.shape[-1]:
                if isinstance(TARGET_COLUMN, str):
                    target_indices = [FEATURE_COLUMNS.index(TARGET_COLUMN)]
                else:  # List of targets
                    target_indices = [FEATURE_COLUMNS.index(tc) for tc in TARGET_COLUMN]
                pt_for_loss = pt[:, :, target_indices]
            else:
                pt_for_loss = pt
            # MAE calculated between denormalized prediction and original target
            test_mae += F.l1_loss(pt_for_loss, yt).item()
    avg_test_mae = test_mae / test_batch_count
    print(f"Test MAE: {avg_test_mae:.6f}")


if __name__ == "__main__":
    main()
