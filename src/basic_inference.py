# inference.py

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime, timedelta

# Import necessary components from provided files
from hidformer import (
    Hidformer,
)  # Assuming hidformer.py is in the same directory or PYTHONPATH
from preprocessing import (
    getData,
)  # Assuming preprocessing.py is in the same directory or PYTHONPATH


def infer_stock(args):
    """
    Performs inference using a trained Hidformer model for a specific stock and date.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure consistent feature columns and target column with training
    # These must match the columns used when the model was trained
    FEATURE_COLUMNS = args.feature_columns or [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]  # Default if not provided
    TARGET_COLUMN = args.target_column  # Must be provided

    if TARGET_COLUMN not in FEATURE_COLUMNS:
        print(
            f"[Error] Target column '{TARGET_COLUMN}' must be included in feature columns: {FEATURE_COLUMNS}"
        )
        return

    input_dim = len(FEATURE_COLUMNS)
    target_col_index = FEATURE_COLUMNS.index(TARGET_COLUMN)
    print(f"Using features: {FEATURE_COLUMNS}")
    print(f"Target column for plotting: {TARGET_COLUMN} (index: {target_col_index})")
    print(f"Input dimension: {input_dim}")

    # --- Data Loading ---
    ticker_safe = args.ticker.replace(".", "_").replace("^", "")
    raw_data_dir = "./data/raw/"  # Define where getData saves/looks for CSVs
    csv_filename = f"{ticker_safe}.csv"
    csv_path = os.path.join(raw_data_dir, csv_filename)

    # Download data if it doesn't exist
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Attempting to download...")
        try:
            getData(args.ticker, output_folder=raw_data_dir, filename=csv_filename)  #
            if not os.path.exists(csv_path):
                print(f"[Error] Failed to download data for {args.ticker}. Exiting.")
                return
        except Exception as e:
            print(
                f"[Error] Exception during data download for {args.ticker}: {e}. Exiting."
            )
            return
    else:
        print(f"Using existing CSV file: {csv_path}")

    # Load the data
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        df.sort_index(inplace=True)
        # Ensure all required feature columns exist
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            print(
                f"[Error] CSV file {csv_path} is missing required feature columns: {missing_cols}. Available: {list(df.columns)}"
            )
            return
        df = df[FEATURE_COLUMNS]  # Keep only necessary columns
        df.dropna(inplace=True)  # Drop rows with any NaN values in selected columns
        if df.empty:
            print(
                f"[Error] DataFrame is empty after loading/filtering/dropping NaNs from {csv_path}. Exiting."
            )
            return
    except Exception as e:
        print(f"[Error] Failed to load or process CSV {csv_path}: {e}")
        return

    # --- Prepare Input Window ---
    try:
        # User specifies the FIRST day of prediction
        prediction_start_date = pd.to_datetime(args.start_date)
        # The input window ENDS the day BEFORE the prediction starts
        # We need to find the latest available data point <= prediction_start_date - 1 day
        # Use business days for indexing potentially
        available_dates = df.index
        # Find the last date in the index that is strictly BEFORE the prediction start date
        potential_end_dates = available_dates[available_dates < prediction_start_date]
        if potential_end_dates.empty:
            print(
                f"[Error] No historical data found before the specified start date {args.start_date}. Cannot form input window."
            )
            return
        input_window_end_date = potential_end_dates[-1]

        print(f"Prediction starts: {prediction_start_date.strftime('%Y-%m-%d')}")
        print(
            f"Input window ends (inclusive): {input_window_end_date.strftime('%Y-%m-%d')}"
        )

        # Select the lookback window data ending on input_window_end_date
        input_window_df = df.loc[:input_window_end_date].tail(args.lookback_window)

        if len(input_window_df) < args.lookback_window:
            print(
                f"[Error] Not enough data ({len(input_window_df)} days) available before {input_window_end_date.strftime('%Y-%m-%d')} to form a lookback window of size {args.lookback_window}."
            )
            return

        print(f"Input window shape (Pandas): {input_window_df.shape}")

        # Convert to NumPy array and then to Tensor
        input_np = input_window_df.values.astype(np.float32)  # (lookback, features)
        input_tensor = (
            torch.tensor(input_np).unsqueeze(0).to(device)
        )  # (1, lookback, features)
        print(f"Input tensor shape: {input_tensor.shape}")

    except Exception as e:
        print(f"[Error] Failed to prepare input window: {e}")
        return

    # --- Prepare Actual Future Data (if available) ---
    actual_data_df = None
    try:
        # Find data starting from prediction_start_date for prediction_horizon days
        actual_data_df = df.loc[prediction_start_date:].head(args.prediction_horizon)
        if len(actual_data_df) > 0:
            print(
                f"Found {len(actual_data_df)} days of actual future data for comparison."
            )
            if len(actual_data_df) < args.prediction_horizon:
                print(
                    f"[Warning] Found only {len(actual_data_df)} days of future data, less than the prediction horizon {args.prediction_horizon}."
                )
        else:
            print(
                "No actual future data found in the loaded CSV for the prediction period."
            )
            actual_data_df = None  # Explicitly set to None if empty
    except Exception as e:
        print(f"[Warning] Could not retrieve or process actual future data: {e}")
        actual_data_df = None

    # --- Model Loading and Inference ---
    if not os.path.exists(args.model_path):
        print(f"[Error] Model checkpoint file not found: {args.model_path}")
        return

    try:
        model = Hidformer(
            input_dim=input_dim,
            pred_len=args.prediction_horizon,
            token_length=args.token_length,
            stride=args.stride,
            num_time_blocks=args.num_time_blocks,
            num_freq_blocks=args.num_freq_blocks,
            d_model=args.d_model,
            freq_k=args.freq_k,
            dropout=args.dropout,
            merge_mode=args.merge_mode,
            merge_k=args.merge_k,
        ).to(
            device
        )  #

        model.load_state_dict(
            torch.load(args.model_path, map_location=device), strict=False
        )
        model.eval()  # Set model to evaluation mode

        print(f"Model loaded successfully from {args.model_path}")

        with torch.no_grad():
            prediction_tensor = model(input_tensor)  # Model output is denormalized

        # Process prediction output
        prediction_np = (
            prediction_tensor.squeeze(0).cpu().numpy()
        )  # (pred_horizon, features)
        predicted_values = prediction_np[
            :, target_col_index
        ]  # Extract target column prediction

        print(f"Prediction tensor shape: {prediction_tensor.shape}")
        print(f"Prediction numpy shape (target): {predicted_values.shape}")

    except Exception as e:
        print(f"[Error] Failed during model loading or inference: {e}")
        return

    # --- Plotting ---
    try:
        # Create date range for the prediction period
        # Use the index from the actual data if available and sufficient, otherwise generate dates
        if actual_data_df is not None and len(actual_data_df) >= len(predicted_values):
            prediction_dates = actual_data_df.index[: len(predicted_values)]
        else:
            # Generate business dates starting from prediction_start_date
            # Adjust length based on the actual prediction output length
            prediction_dates = pd.bdate_range(
                start=prediction_start_date, periods=len(predicted_values)
            )
            if len(prediction_dates) == 0:
                print(
                    "[Error] Could not generate prediction dates. Check start date and prediction horizon."
                )
                return

        plt.figure(figsize=(12, 6))
        plt.plot(
            prediction_dates,
            predicted_values,
            label=f"Predicted {TARGET_COLUMN}",
            marker="o",
            linestyle="--",
        )

        if actual_data_df is not None:
            actual_values_plot = actual_data_df[TARGET_COLUMN].iloc[
                : len(predicted_values)
            ]  # Ensure length match
            if not actual_values_plot.empty:
                plt.plot(
                    prediction_dates[: len(actual_values_plot)],
                    actual_values_plot,
                    label=f"Actual {TARGET_COLUMN}",
                    marker="x",
                    linestyle="-",
                )

        plt.title(
            f'{args.ticker} - {TARGET_COLUMN} Prediction vs Actual\n(Prediction Start: {prediction_start_date.strftime("%Y-%m-%d")})'
        )
        plt.xlabel("Date")
        plt.ylabel(TARGET_COLUMN)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Ensure plot only spans the prediction window
        plt.xlim(prediction_dates.min(), prediction_dates.max())

        plt.show()

    except Exception as e:
        print(f"[Error] Failed during plotting: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidformer Inference Script")

    # Required arguments
    parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL)"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="First date of the prediction window (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="Close",
        help="Name of the target column to predict and plot (must be in features)",
    )

    # Data arguments (should match training)
    parser.add_argument(
        "--lookback_window", type=int, default=128, help="Lookback window size (T_in)"
    )
    parser.add_argument(
        "--prediction_horizon", type=int, default=128, help="Prediction horizon (H)"
    )
    parser.add_argument(
        "--feature_columns",
        nargs="+",
        default=["Open", "High", "Low", "Close", "Volume"],
        help="List of feature columns used during training",
    )

    # Model hyperparameters (should match the loaded model's training config)
    parser.add_argument(
        "--token_length", type=int, default=32, help="Token length for segmentation"
    )  #
    parser.add_argument(
        "--stride", type=int, default=16, help="Stride for segmentation"
    )  #
    parser.add_argument(
        "--num_time_blocks", type=int, default=4, help="Number of TimeBlocks"
    )  #
    parser.add_argument(
        "--num_freq_blocks", type=int, default=2, help="Number of FrequencyBlocks"
    )  #
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")  #
    parser.add_argument(
        "--freq_k", type=int, default=64, help="Low-rank dimension for LinearAttention"
    )  #
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")  #
    parser.add_argument(
        "--merge_mode",
        type=str,
        default="linear",
        choices=["linear", "mean"],
        help="Mergence layer mode",
    )  #
    parser.add_argument(
        "--merge_k",
        type=int,
        default=2,
        help="Number of tokens to merge in MergenceLayer",
    )  #

    args = parser.parse_args()

    # Basic validation for date format
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
    except ValueError:
        print("[Error] start_date must be in YYYY-MM-DD format.")
        exit()

    infer_stock(args)
