import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np


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
                            Default: 5 (as mentioned in Hidformer paper)
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def main():
    # --- Configuration ---
    TICKER_LIST_CSV = "./src/tickers.csv"  # Path to your ticker list
    DATA_ROOT_DIR = "./data/"  # Root directory for raw/processed data
    MODEL_SAVE_PATH = "./model/test_larger_dataset.pt"  # Path to save best model

    # Data parameters
    LOOKBACK_WINDOW = 120  # Lookback window (Hidformer input length T_in)
    PREDICTION_HORIZON = 10  # Forecast horizon (Hidformer pred_len H)
    # Features to use (ensure these match columns in downloaded CSVs from getData)
    # Default OHLCV from preprocessing.py
    FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
    TARGET_COLUMN = "Close"  # Target column for prediction (can be a list too)

    # Model parameters (match Hidformer definition)
    TOKEN_LENGTH = 24
    STRIDE = 12
    NUM_TIME_BLOCKS = 4
    NUM_FREQ_BLOCKS = 2
    D_MODEL = 128
    FREQ_K = 64
    DROPOUT = 0.2
    MERGE_MODE = "linear"
    MERGE_K = 2

    # Training parameters
    BATCH_SIZE = 32
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
        ticker_df = pd.read_csv(TICKER_LIST_CSV)
        tickers = ticker_df.iloc[:, 0].astype(str).unique().tolist()
        print(f"Read {len(tickers)} tickers from {TICKER_LIST_CSV}")
    except Exception as e:
        print(f"Error reading ticker CSV {TICKER_LIST_CSV}: {e}")
        return

    # Download data if needed (using preprocessing.getData)
    # This is handled within createDataset if download=True
    # You might want to run getData separately first if needed:
    # for ticker in tickers:
    #     getData(ticker, output_folder=os.path.join(DATA_ROOT_DIR, 'raw'))

    # Create datasets using preprocessing.createDataset
    try:
        print("Creating train dataset...")
        train_ds = createDataset(
            ticker_list_csv_path=TICKER_LIST_CSV,
            root=DATA_ROOT_DIR,
            lookback_window=LOOKBACK_WINDOW,
            prediction_horizon=PREDICTION_HORIZON,
            split="train",
            target_column=TARGET_COLUMN,
            feature_columns=FEATURE_COLUMNS,
            download=True,  # Set to True to download if raw files missing
            force_regenerate=False,  # Set to True to force reprocessing
        )
        print("Creating validation dataset...")
        val_ds = createDataset(
            ticker_list_csv_path=TICKER_LIST_CSV,
            root=DATA_ROOT_DIR,
            lookback_window=LOOKBACK_WINDOW,
            prediction_horizon=PREDICTION_HORIZON,
            split="val",
            target_column=TARGET_COLUMN,
            feature_columns=FEATURE_COLUMNS,
            download=False,  # Assumes data downloaded during train split creation
            force_regenerate=False,
        )
        print("Creating test dataset...")
        test_ds = createDataset(
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

    # Create DataLoaders
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
        pred_len=PREDICTION_HORIZON,  # Pass prediction horizon
        token_length=TOKEN_LENGTH,
        stride=STRIDE,
        num_time_blocks=NUM_TIME_BLOCKS,  # Use updated arg name
        num_freq_blocks=NUM_FREQ_BLOCKS,  # Use updated arg name
        d_model=D_MODEL,
        freq_k=FREQ_K,
        dropout=DROPOUT,
        merge_mode=MERGE_MODE,
        merge_k=MERGE_K,
        # Removed hidden_size (SRUpp defaults to d_model)
        # Removed out_dim (calculated internally by decoder)
    ).to(device)
    print(
        f"Model instantiated with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    # --- Loss and Optimizer ---
    criterion = nn.MSELoss()  # Use MSE as per Hidformer paper
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
        for X, y in train_loader:
            # X: (B, LOOKBACK_WINDOW, input_dim) from createDataset
            # y: (B, PREDICTION_HORIZON, num_targets) from createDataset
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # Model forward pass
            pred = model(X)  # Output: (B, PREDICTION_HORIZON, input_dim)

            # --- Loss Calculation ---
            # Ensure target 'y' has the correct shape and features if needed
            # If target_column is just 'Close', y might be (B, H, 1)
            # If model predicts all features (B, H, C), need to select target from pred
            if y.shape[-1] != pred.shape[-1]:
                # Assuming y contains only the target column(s)
                # And model predicts all input features
                # Find index of target column in FEATURE_COLUMNS
                if isinstance(TARGET_COLUMN, str):
                    target_indices = [FEATURE_COLUMNS.index(TARGET_COLUMN)]
                else:  # List of targets
                    target_indices = [FEATURE_COLUMNS.index(tc) for tc in TARGET_COLUMN]

                if len(target_indices) != y.shape[-1]:
                    raise ValueError(
                        f"Shape mismatch: Target y has {y.shape[-1]} features, but found {len(target_indices)} target indices."
                    )

                pred_for_loss = pred[:, :, target_indices]  # Select predicted target(s)
            else:
                # Assume y contains all features, matching prediction
                pred_for_loss = pred

            # Check shapes before loss
            if pred_for_loss.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch before loss: pred={pred_for_loss.shape}, target={y.shape}"
                )

            loss = criterion(pred_for_loss, y)
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
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                pv = model(Xv)  # Output: (B, PREDICTION_HORIZON, input_dim)

                # Select target features from prediction if necessary
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

                val_loss += criterion(pv_for_loss, yv).item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        print(
            f"Epoch {epoch:02d}/{EPOCHS} — Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # --- Early Stopping Check ---
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\n--- Training Finished ---")

    # --- Testing Loop ---
    print("\n--- Starting Testing ---")
    # Load the best model saved by early stopping
    try:
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
        for Xt, yt in test_loader:
            Xt, yt = Xt.to(device), yt.to(device)
            pt = model(Xt)  # Output: (B, PREDICTION_HORIZON, input_dim)

            # Select target features from prediction if necessary
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

            test_loss += criterion(pt_for_loss, yt).item()
            test_batch_count += 1

    avg_test_loss = test_loss / test_batch_count
    print(f"Test MSE: {avg_test_loss:.6f}")

    # --- Optional: Add MAE calculation for testing ---
    test_mae = 0.0
    with torch.no_grad():
        for Xt, yt in test_loader:
            Xt, yt = Xt.to(device), yt.to(device)
            pt = model(Xt)
            if yt.shape[-1] != pt.shape[-1]:
                if isinstance(TARGET_COLUMN, str):
                    target_indices = [FEATURE_COLUMNS.index(TARGET_COLUMN)]
                else:  # List of targets
                    target_indices = [FEATURE_COLUMNS.index(tc) for tc in TARGET_COLUMN]
                pt_for_loss = pt[:, :, target_indices]
            else:
                pt_for_loss = pt
            test_mae += F.l1_loss(pt_for_loss, yt).item()  # Use L1 loss for MAE
    avg_test_mae = test_mae / test_batch_count
    print(f"Test MAE: {avg_test_mae:.6f}")


# def main():
#     # 2.1 — Configure & prepare data

#     tickers = ["MSFT", "AAPL", "NVDA", "TSM"]
#     seq_len = 60  # lookback window
#     pred_len = 10  # forecast horizon
#     batch_sz = 32
#     dp = DataPipeline(
#         tickers=tickers,
#         start_date="2020-01-01",
#         end_date="2023-12-31",
#         seq_len=seq_len,
#         pred_len=pred_len,
#         val_ratio=0.1,
#         test_ratio=0.1,
#         batch_size=batch_sz,
#     )
#     dp.prepare_data()
#     train_loader, val_loader, test_loader = dp.get_loaders()
#     print(
#         f"Train samples: {len(dp.train_ds)}, Val: {len(dp.val_ds)}, Test: {len(dp.test_ds)}"
#     )

#     # 2.2 — Instantiate model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # input_dim = number of tickers = 4
#     model = Hidformer(
#         input_dim=len(tickers),
#         token_length=16,
#         stride=8,
#         time_blocks=4,
#         freq_blocks=2,
#         hidden_size=128,
#         freq_k=64,
#         out_dim=len(tickers) * pred_len,  # predict pred_len steps for each ticker
#     ).to(device)

#     # 2.3 — Loss and optimizer
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)

#     # 2.4 — Training loop
#     epochs = 20
#     for epoch in range(1, epochs + 1):
#         model.train()
#         running_loss = 0.0
#         for X, y in train_loader:
#             # X: (B, seq_len, 4),  y: (B, pred_len, 4)
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             pred = model(X)
#             # pred: (B, 4*pred_len) — reshape to match y
#             pred = pred.view(-1, pred_len, len(tickers))
#             loss = criterion(pred, y)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * X.size(0)

#         train_loss = running_loss / len(dp.train_ds)

#         # 2.5 — Validation
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for Xv, yv in val_loader:
#                 Xv, yv = Xv.to(device), yv.to(device)
#                 pv = model(Xv).view(-1, pred_len, len(tickers))
#                 val_loss += criterion(pv, yv).item() * Xv.size(0)
#         val_loss /= len(dp.val_ds)

#         print(f"Epoch {epoch:02d} — Train: {train_loss:.4f}, Val: {val_loss:.4f}")

#     # 2.6 — Test set
#     test_loss = 0.0
#     with torch.no_grad():
#         for Xt, yt in test_loader:
#             Xt, yt = Xt.to(device), yt.to(device)
#             pt = model(Xt).view(-1, pred_len, len(tickers))
#             test_loss += criterion(pt, yt).item() * Xt.size(0)
#     test_loss /= len(dp.test_ds)
#     print(f"Test MSE: {test_loss:.4f}")


if __name__ == "__main__":
    main()
