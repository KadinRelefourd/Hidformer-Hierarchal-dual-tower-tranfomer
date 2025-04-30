import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import argparse

# Assume hidformer.py is in the same directory or accessible via PYTHONPATH
try:
    from hidformer import Hidformer
except ImportError:
    print(
        "Error: hidformer.py not found. Make sure it's in the same directory or your PYTHONPATH."
    )
    exit()

# --- Default Configuration ---
DEFAULT_MODEL_PATH = "./model/test_larger_dataset.pt"
DEFAULT_TICKER = "ATUS"
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2024-01-01"  # Predict up to this date

# --- Model & Data Parameters (MUST match training) ---
LOOKBACK_WINDOW = 128
PREDICTION_HORIZON = 10  # From training
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]  # From training
TARGET_COLUMN = "Close"  # From training
INPUT_DIM = len(FEATURE_COLUMNS)

# Model Hyperparameters (MUST match the trained model)
TOKEN_LENGTH = 24
STRIDE = 12
NUM_TIME_BLOCKS = 4
NUM_FREQ_BLOCKS = 2
D_MODEL = 128
FREQ_K = 64
DROPOUT = 0.2
MERGE_MODE = "linear"
MERGE_K = 2

# --- Helper Functions ---


def download_stock_data(ticker, start, end, feature_cols):
    """
    Downloads historical stock data using yfinance, handling potential
    multi-level columns similarly to preprocessing.py.

    Args:
        ticker (str): The ticker symbol.
        start (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).
        feature_cols (list): List of required feature column names.

    Returns:
        pd.DataFrame or None: DataFrame with Date index and feature_cols,
                              or None if download/processing fails.
    """
    print(f"--- Downloading data for: {ticker} ({start} to {end}) ---")
    try:
        # Use auto_adjust=True consistent with preprocessing.getData
        ticker_data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,  # Gets adjusted OHLC, Volume might be unadjusted/missing
            # group_by="ticker", # Not needed for single ticker download
        )

        if ticker_data.empty:
            print(f"[Warning] No data found for {ticker}.")
            return None

        # --- Handle Potential MultiIndex Columns (based on preprocessing.py) ---
        if isinstance(ticker_data.columns, pd.MultiIndex):
            print(f"[Info] Detected multi-level columns for {ticker}. Flattening.")
            # Assumes the desired column names ('Open', 'High', etc.) are in the first level (0)
            ticker_data.columns = ticker_data.columns.get_level_values(0)
            # Remove duplicate columns potentially created by flattening
            ticker_data = ticker_data.loc[
                :, ~ticker_data.columns.duplicated(keep="first")
            ]
            print(f"[Info] Columns after flattening: {list(ticker_data.columns)}")

        # --- Ensure Date is Index ---
        # yf.download usually returns Date as index, but check just in case
        if not isinstance(ticker_data.index, pd.DatetimeIndex):
            if "Date" in ticker_data.columns:
                print("[Info] Setting 'Date' column as index.")
                ticker_data["Date"] = pd.to_datetime(ticker_data["Date"])
                ticker_data.set_index("Date", inplace=True)
            else:
                # This case is unlikely with yfinance but could happen if format changes
                print("[Error] Cannot find 'Date' index or column.")
                return None

        # --- Check Required Columns ---
        available_columns = list(ticker_data.columns)
        missing_cols = [col for col in feature_cols if col not in available_columns]

        # Handle specifically missing 'Volume' if it's the only one missing
        if "Volume" in missing_cols and len(missing_cols) == 1:
            print("[Warning] Filling missing 'Volume' column with 0.")
            ticker_data["Volume"] = 0.0  # Add volume column filled with zeros
            missing_cols.remove("Volume")  # Remove volume from missing list

        # Check if any other essential columns are missing
        if missing_cols:
            print(
                f"[Error] Missing essential columns: {missing_cols}. Available: {available_columns}"
            )
            return None

        # --- Select and Reorder Columns ---
        # Ensure the DataFrame contains only the required features in the correct order
        try:
            ticker_data_selected = ticker_data[feature_cols]
        except KeyError as e:
            print(
                f"[Error] Could not select all required feature columns: {e}. Available: {list(ticker_data.columns)}"
            )
            return None

        print(
            f"Downloaded and processed {len(ticker_data_selected)} rows. Columns: {list(ticker_data_selected.columns)}"
        )
        return ticker_data_selected

    except Exception as e:
        print(f"[Error] Data download/processing failed unexpectedly: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback for debugging
        return None


def create_inference_windows(df, feature_cols, lookback):  # <<< Remove scaler argument
    """Creates all possible lookback windows from raw data."""
    print("Creating inference windows from raw data...")
    if df is None or len(df) < lookback:
        print(
            f"Error: Not enough data ({len(df) if df is not None else 0} rows) for lookback ({lookback})."
        )
        return None, None

    # Select feature columns BUT DO NOT SCALE
    data_raw = df[feature_cols]
    print(f"Using raw features: {feature_cols}")

    # Create all possible sliding windows from raw data
    X, sequence_end_dates = [], []
    for i in range(len(data_raw) - lookback + 1):
        sequence = data_raw.iloc[i : i + lookback].values  # Use raw values
        X.append(sequence)
        sequence_end_dates.append(data_raw.index[i + lookback - 1])

    if not X:
        print("Error: No sequences created.")
        return None, None

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    sequence_end_dates_idx = pd.DatetimeIndex(sequence_end_dates)

    print(f"Created {X_tensor.shape[0]} sequences for inference (unscaled).")
    return X_tensor, sequence_end_dates_idx


def run_inference(model, data_tensor, device, batch_size=64):
    """Runs the model in evaluation mode to get predictions.
    Assumes model output is DENORMALIZED by its internal RevIN layer."""
    print(f"Running inference on {data_tensor.shape[0]} sequences...")
    model.eval()
    predictions = []
    with torch.no_grad():
        num_sequences = data_tensor.shape[0]
        for i in range(0, num_sequences, batch_size):
            batch_X = data_tensor[i : min(i + batch_size, num_sequences)].to(device)
            # Model takes SCALED input, output is assumed DENORMALIZED by RevIN
            batch_pred = model(batch_X)  # Shape: (batch, pred_horizon, features)
            predictions.append(batch_pred.cpu().numpy())

    predictions_np = np.concatenate(predictions, axis=0)
    print("Inference complete.")
    print(
        f"Shape of predictions array: {predictions_np.shape}"
    )  # (total_sequences, pred_horizon, features)
    # Basic check for issues
    if np.isnan(predictions_np).any() or np.isinf(predictions_np).any():
        print("[ERROR] Predictions contain NaN or Inf values!")
    else:
        print(
            f"Prediction range (all features/horizons): min={np.min(predictions_np):.2f}, max={np.max(predictions_np):.2f}"
        )
    return predictions_np


def plot_predictions_vs_actual(
    actual_df,
    predicted_data,  # Full prediction array (num_sequences, pred_horizon, num_features)
    sequence_end_dates,  # Dates corresponding to the *end* of each input sequence
    ticker,
    target_col_name,
    feature_cols,
    plot_horizon=1,  # Which prediction step ahead to plot (e.g., 1 = next day)
):
    """Plots actual vs predicted prices for a specific horizon."""
    print(f"Visualizing forecast for horizon H={plot_horizon}...")

    if plot_horizon < 1 or plot_horizon > predicted_data.shape[1]:
        print(
            f"Error: plot_horizon ({plot_horizon}) out of range (1 to {predicted_data.shape[1]})"
        )
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(14, 7))

    # --- Get Actual Data ---
    actual_series = actual_df[target_col_name]

    # --- Get Predicted Data for the specific horizon ---
    try:
        target_col_idx = feature_cols.index(target_col_name)
    except ValueError:
        print(
            f"Error: Target column '{target_col_name}' not found in feature_cols: {feature_cols}"
        )
        return

    # predicted_data shape: (num_sequences, pred_horizon, num_features)
    # We want the prediction made *after* each sequence_end_date, for the specific horizon.
    preds_h = predicted_data[
        :, plot_horizon - 1, target_col_idx
    ]  # Shape: (num_sequences,)

    # --- Align Dates ---
    # The prediction `preds_h[i]` corresponds to `plot_horizon` steps *after* `sequence_end_dates[i]`.
    # We need to find the actual calendar dates for these predictions.
    pred_dates = []
    valid_preds_h = []
    actual_dates_index = actual_df.index

    for i, seq_end_date in enumerate(sequence_end_dates):
        # Find the position of the sequence end date in the original dataframe index
        try:
            # searchsorted finds insertion point; if date exists, loc will be its index
            loc = actual_dates_index.get_loc(seq_end_date)
            # Target date is plot_horizon business days *after* loc
            target_date_loc = loc + plot_horizon
            if target_date_loc < len(actual_dates_index):
                pred_dates.append(actual_dates_index[target_date_loc])
                valid_preds_h.append(preds_h[i])
            # else: prediction date is beyond the actual data range
        except KeyError:
            # If seq_end_date isn't in index (e.g., weekend/holiday), find next valid date
            # This logic might need refinement depending on how you want to handle non-trading days
            later_dates = actual_dates_index[actual_dates_index > seq_end_date]
            if len(later_dates) >= plot_horizon:
                pred_dates.append(later_dates[plot_horizon - 1])
                valid_preds_h.append(preds_h[i])
            # else: prediction date is beyond the actual data range

    if not pred_dates:
        print("Warning: Could not determine valid dates for plotting predictions.")
        # Plot only actual data if no predictions can be mapped
        plt.plot(
            actual_series.index,
            actual_series,
            label=f"Actual {target_col_name}",
            color="royalblue",
            linewidth=2,
        )
    else:
        pred_series = pd.Series(valid_preds_h, index=pd.DatetimeIndex(pred_dates))
        # Plot both, aligning them on the date axis
        plt.plot(
            actual_series.index,
            actual_series,
            label=f"Actual {target_col_name}",
            color="royalblue",
            linewidth=2,
        )
        plt.plot(
            pred_series.index,
            pred_series,
            label=f"Predicted {target_col_name} (H={plot_horizon})",
            color="darkorange",
            linestyle="--",
            marker=".",
            markersize=4,
        )
        print(
            f"Plotting {len(pred_series)} predictions against {len(actual_series)} actual points."
        )
        # Print some diagnostics
        print(f"Actual Min: {actual_series.min():.2f}, Max: {actual_series.max():.2f}")
        if not pred_series.empty:
            if pred_series.isnull().any() or np.isinf(pred_series).any():
                print(
                    f"[ERROR] Predicted series for H={plot_horizon} contains NaN/Inf!"
                )
            else:
                print(
                    f"Predicted Min: {pred_series.min():.2f}, Max: {pred_series.max():.2f}"
                )
        else:
            print("Predicted series is empty.")

    plt.title(
        f"{ticker} Stock Price - Actual vs Predicted (Horizon={plot_horizon} days)",
        fontsize=16,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_single_window_forecast(
    actual_df,  # DataFrame with actual prices (Date index)
    predictions_denorm,  # Full prediction array (sequences, horizon, features)
    sequence_end_dates,  # DatetimeIndex of the end date for each input sequence
    window_index_to_plot,  # Index of the sequence/prediction to plot (e.g., -1 for last)
    ticker,  # Ticker symbol (for title)
    target_col_name,  # Name of the target column (e.g., 'Close')
    feature_cols,  # List of all feature names
    pred_horizon,  # The prediction horizon (e.g., 10)
):
    """Plots the multi-step forecast from a single input window vs actual data."""
    print(
        f"\n--- Visualizing forecast for single window (index: {window_index_to_plot}) ---"
    )

    num_sequences = predictions_denorm.shape[0]
    if not (-num_sequences <= window_index_to_plot < num_sequences):
        print(
            f"Error: window_index_to_plot ({window_index_to_plot}) is out of bounds for {num_sequences} sequences."
        )
        return

    # --- Get the specific prediction sequence ---
    # shape: (pred_horizon, num_features)
    single_prediction = predictions_denorm[window_index_to_plot]

    # --- Get the predicted target values ---
    try:
        target_col_idx = feature_cols.index(target_col_name)
    except ValueError:
        print(
            f"Error: Target column '{target_col_name}' not found in feature_cols: {feature_cols}"
        )
        return
    # shape: (pred_horizon,)
    predicted_target_values = single_prediction[:, target_col_idx]

    # --- Determine the dates for the prediction period ---
    input_sequence_end_date = sequence_end_dates[window_index_to_plot]
    print(f"Input window ended on: {input_sequence_end_date.strftime('%Y-%m-%d')}")

    # Find the actual dates in the DataFrame that correspond to the prediction horizon
    actual_dates_index = actual_df.index
    # Find dates strictly *after* the input sequence ended
    potential_prediction_dates = actual_dates_index[
        actual_dates_index > input_sequence_end_date
    ]

    if len(potential_prediction_dates) < pred_horizon:
        print(
            f"Warning: Only found {len(potential_prediction_dates)} actual dates after {input_sequence_end_date.strftime('%Y-%m-%d')} (needed {pred_horizon}). Plotting available data."
        )
        prediction_dates = potential_prediction_dates
        # Trim predictions if actual data is shorter
        predicted_target_values = predicted_target_values[: len(prediction_dates)]
    else:
        # Select the next `pred_horizon` available dates
        prediction_dates = potential_prediction_dates[:pred_horizon]

    if prediction_dates.empty:
        print(
            "Error: Could not find any actual dates following the input sequence end date."
        )
        return

    print(
        f"Plotting prediction for dates: {prediction_dates[0].strftime('%Y-%m-%d')} to {prediction_dates[-1].strftime('%Y-%m-%d')}"
    )

    # --- Get the actual target values for the prediction dates ---
    try:
        actual_target_values = actual_df.loc[prediction_dates, target_col_name]
    except KeyError:
        print(
            f"Error: Could not retrieve actual values for the determined prediction dates."
        )
        return

    # --- Create the plot ---
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10, 6))

    plt.plot(
        prediction_dates,
        actual_target_values.values,
        label=f"Actual {target_col_name}",
        color="royalblue",
        marker="o",
        linewidth=2,
    )
    plt.plot(
        prediction_dates,
        predicted_target_values,
        label=f"Predicted {target_col_name} (Horizon 1-{len(prediction_dates)})",
        color="darkorange",
        marker="x",
        linestyle="--",
        linewidth=2,
    )

    plt.title(
        f'{ticker} Forecast vs Actual ({pred_horizon}-Day Horizon)\nInput Window Ended: {input_sequence_end_date.strftime("%Y-%m-%d")}',
        fontsize=14,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and plot predictions.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model (.pt file).",
    )
    parser.add_argument(
        "--ticker", type=str, default=DEFAULT_TICKER, help="Stock ticker symbol."
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Start date for fetching data (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=DEFAULT_END_DATE,
        help="End date for fetching data (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--plot_horizon",
        type=int,
        default=1,
        help="Which prediction horizon step to plot (e.g., 1 for 1 day ahead).",
    )

    args = parser.parse_args()

    # --- Validate Inputs ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit()
    if args.plot_horizon < 1 or args.plot_horizon > PREDICTION_HORIZON:
        print(
            f"Error: plot_horizon ({args.plot_horizon}) must be between 1 and {PREDICTION_HORIZON}."
        )
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from {args.model_path}...")
    try:
        model = Hidformer(
            input_dim=INPUT_DIM,
            pred_len=PREDICTION_HORIZON,  # Must match trained model
            token_length=TOKEN_LENGTH,
            stride=STRIDE,
            num_time_blocks=NUM_TIME_BLOCKS,
            num_freq_blocks=NUM_FREQ_BLOCKS,
            d_model=D_MODEL,
            freq_k=FREQ_K,
            dropout=DROPOUT,  # Dropout doesn't affect inference if model.eval() is called
            merge_mode=MERGE_MODE,
            merge_k=MERGE_K,
        ).to(device)
        # Load state dict - use map_location for flexibility
        model.load_state_dict(
            torch.load(args.model_path, map_location=device), strict=False
        )
        model.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure architecture parameters match the saved model.")
        exit()

    # --- Get Data ---
    # Fetch slightly more data initially if possible, to ensure enough for the first lookback window
    # Example: Fetch data starting LOOKBACK_WINDOW days earlier if start_date is the first prediction point.
    # For simplicity here, we just fetch the user-specified range.
    df_actual = download_stock_data(
        args.ticker, args.start_date, args.end_date, FEATURE_COLUMNS
    )
    if df_actual is None:
        print("Exiting: Failed to download data.")
        exit()

    # --- Preprocess Data (Create unscaled Windows) ---
    X_inference_tensor, sequence_end_dates = create_inference_windows(
        df_actual, FEATURE_COLUMNS, LOOKBACK_WINDOW
    )
    if X_inference_tensor is None:
        print("Exiting: Failed to create inference windows.")
        exit()

    # --- Run Inference ---
    # Output `predictions_denorm` is assumed denormalized by the model's RevIN layer
    predictions_denorm = run_inference(model, X_inference_tensor, device)
    if predictions_denorm is None:
        print("Exiting: Inference failed.")
        exit()

    # --- Visualize ---
    plot_predictions_vs_actual(
        actual_df=df_actual,
        predicted_data=predictions_denorm,
        sequence_end_dates=sequence_end_dates,
        ticker=args.ticker,
        target_col_name=TARGET_COLUMN,
        feature_cols=FEATURE_COLUMNS,
        plot_horizon=args.plot_horizon,
    )

    plot_single_window_forecast(
        actual_df=df_actual,
        predictions_denorm=predictions_denorm,
        sequence_end_dates=sequence_end_dates,
        window_index_to_plot=0,  # Use -1 for the last window, 0 for first, etc.
        ticker=args.ticker,
        target_col_name=TARGET_COLUMN,
        feature_cols=FEATURE_COLUMNS,
        pred_horizon=PREDICTION_HORIZON,  # Should be 10 based on your setup
    )
    print("\n--- Inference and Plotting Complete ---")
