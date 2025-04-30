import torch
import yfinance as yf
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import hashlib


def create_sliding_windows(
    csv_path,
    lookback_window,
    prediction_horizon,
    target_column="Close",
    feature_columns=None,
):
    """
    Reads a time series CSV and creates a dataset of sliding windows.

    Args:
        csv_path (str): Path to the input CSV file.
                        Expected columns: 'Date' and feature/target columns.
        lookback_window (int): The number of time steps to include in
                               each input sequence (X).
        prediction_horizon (int): The number of time steps to predict
                                  into the future (y).
        target_column (str or list): The name(s) of the column(s) to predict.
                                     Defaults to 'Close'.
        feature_columns (list, optional): A list of column names to use as
                                          input features. If None, all columns
                                          except 'Date' and the target_column(s)
                                          are used. Defaults to None.

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - X_data (np.ndarray): Input sequences (samples, lookback_window, num_features).
               - y_data (np.ndarray): Target sequences (samples, prediction_horizon, num_targets).
               Returns (None, None) if an error occurs or not enough data.
    """
    print(f"--- Creating windows for: {csv_path} ---")
    # --- 1. Load Data ---
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found: {csv_path}")
        return None, None
    try:
        # Read CSV, parse 'Date' column
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        # Sort by date just in case it's not already sorted
        df.sort_index(inplace=True)
    except Exception as e:
        print(f"[Error] Failed to read or parse CSV {csv_path}: {e}")
        return None, None

    # --- 2. Identify Features and Target ---
    if isinstance(target_column, str):
        target_column_list = [target_column]
    else:
        target_column_list = list(target_column)

    if feature_columns is None:
        # Default: Use all columns as features initially
        feature_columns = list(df.columns)
        # Remove target columns from features if they overlap
        feature_columns = [
            col for col in feature_columns if col not in target_column_list
        ]
    else:
        # Ensure provided feature columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            print(
                f"[Error] Specified feature columns not found in CSV: {missing_features}"
            )
            return None, None

    # Ensure target columns exist
    missing_targets = [col for col in target_column_list if col not in df.columns]
    if missing_targets:
        print(f"[Error] Specified target columns not found in CSV: {missing_targets}")
        return None, None

    print(f"Using features: {feature_columns}")
    print(f"Predicting target(s): {target_column_list}")

    # Select only necessary columns
    feature_df = df[feature_columns]
    target_df = df[target_column_list]

    # --- 3. Check Data Length ---
    total_length = len(df)
    required_length = lookback_window + prediction_horizon
    if total_length < required_length:
        print(
            f"[Error] Not enough data ({total_length} rows) to create even one window "
            f"with lookback={lookback_window} and horizon={prediction_horizon}."
        )
        return None, None

    # --- 4. Generate Windows ---
    X_list = []
    y_list = []

    # Iterate through the DataFrame to create windows
    # The last possible start index for the lookback window is: total_length - lookback_window - prediction_horizon
    for i in range(total_length - required_length + 1):
        # Input window (X): from index i to i + lookback_window
        input_window_end = i + lookback_window
        X_window = feature_df.iloc[i:input_window_end].values

        # Target window (y): from index input_window_end to input_window_end + prediction_horizon
        target_window_end = input_window_end + prediction_horizon
        y_window = target_df.iloc[input_window_end:target_window_end].values

        X_list.append(X_window)
        y_list.append(y_window)

    # Convert lists to NumPy arrays
    X_data = np.array(X_list)
    y_data = np.array(y_list)

    print(f"Created dataset with {X_data.shape[0]} windows.")
    print(
        f"Input shape (X): {X_data.shape}"
    )  # (num_samples, lookback_window, num_features)
    print(
        f"Target shape (y): {y_data.shape}"
    )  # (num_samples, prediction_horizon, num_targets)

    return X_data, y_data


def getData(ticker_symbol, output_folder="./data/csv/", filename=None):
    """
    Downloads historical market data (adjusted OHLC and Volume) for a
    given ticker symbol using yfinance and saves it to a CSV file with the
    header reflecting available columns (typically Date,Open,High,Low,Close,Volume).

    Uses auto_adjust=True to get split/dividend adjusted OHLC data.
    Note: Volume is typically not adjusted by yfinance.

    Args:
        ticker_symbol (str): The ticker symbol of the stock (e.g., 'AAPL').
        output_folder (str): The folder where the CSV file will be saved.
                             Defaults to './data/csv/'.
        filename (str, optional): The desired name for the output CSV file.
                                  If None, defaults to '{ticker_symbol}_adj_ohlcv.csv'.
                                  Defaults to None.
    """
    print(f"--- Attempting to download data for: {ticker_symbol} ---")
    try:
        # Download historical data using yfinance
        # Set auto_adjust=True (or omit, as it's default) to get
        # OHLC data adjusted for splits and dividends.
        # Note: This might remove the 'Volume' column for some assets or versions.
        ticker_data = yf.download(
            ticker_symbol,
            period="max",
            progress=False,
            auto_adjust=True,  # Get adjusted OHLC
        )
        print(ticker_data.head())  # Debug: Show first few rows of the downloaded data
        # --- Handle Potential MultiIndex Columns ---
        # Although less common with auto_adjust=True, check just in case.
        if isinstance(ticker_data.columns, pd.MultiIndex):
            print(
                f"[Info] Detected multi-level columns for {ticker_symbol}. Flattening."
            )
            ticker_data.columns = ticker_data.columns.get_level_values(0)
            ticker_data = ticker_data.loc[
                :, ~ticker_data.columns.duplicated(keep="first")
            ]

        # Check if the downloaded data is empty
        if ticker_data.empty:
            print(f"[Warning] No data found for {ticker_symbol}. Skipping.")
            return

        # --- Data Selection ---
        # Reset the index to make 'Date' a regular column
        ticker_data.reset_index(inplace=True)

        # Define the desired columns (adjusted OHLC + V)
        # Note: 'Adj Close' is usually not present or is redundant when auto_adjust=True
        # Note: 'Volume' might be missing depending on yfinance version/asset type with auto_adjust=True
        desired_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        # Select only the desired columns that ACTUALLY EXIST in the downloaded data
        available_columns = [
            col for col in desired_columns if col in ticker_data.columns
        ]
        print(
            f"[Info] Columns available after download for {ticker_symbol}: {list(ticker_data.columns)}"
        )
        print(f"[Info] Selecting columns for CSV: {available_columns}")

        if "Date" not in available_columns:
            print(
                f"[Error] 'Date' column is unexpectedly missing for {ticker_symbol}. Skipping save."
            )
            return
        if not any(
            col in available_columns for col in ["Open", "High", "Low", "Close"]
        ):
            print(f"[Error] No OHLC columns found for {ticker_symbol}. Skipping save.")
            return

        ticker_data_final = ticker_data[available_columns]

        # --- File and Folder Handling ---
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                print(f"Created output folder: {output_folder}")
            except OSError as e:
                print(
                    f"[Error] Could not create output folder {output_folder}: {e}. Saving to current directory instead."
                )
                output_folder = "."

        if filename is None:
            # Default filename format: TICKER_adj_ohlcv.csv
            output_filename = f"{ticker_symbol.replace('.', '_')}.csv"
        else:
            # Use provided filename, ensuring it ends with .csv
            if not filename.lower().endswith(".csv"):
                output_filename = f"{filename}.csv"
            else:
                output_filename = filename

        # Construct the full file path
        output_path = os.path.join(output_folder, output_filename)

        # --- Saving the Data ---
        # Save the selected DataFrame (Adjusted OHLC + Volume if available) to CSV
        # index=False: Do not write the DataFrame index as a column
        # header=True: Write the available column names as the header
        ticker_data_final.to_csv(output_path, index=False, header=True)

        print(
            f"[Success] Successfully downloaded and saved Adjusted OHLCV data for {ticker_symbol} to {output_path}"
        )
        print(f"CSV Header Written: {','.join(ticker_data_final.columns)}")

    except Exception as e:
        # Catch any other unexpected exceptions
        print(
            f"[Error] An unexpected error occurred while processing ticker {ticker_symbol}: {e}"
        )


def createDataset(
    ticker_list_csv_path,
    root,
    lookback_window,
    prediction_horizon,
    split="train",
    target_column="Close",
    feature_columns=None,
    download=False,
    split_percentages=(0.7, 0.15, 0.15),
    force_regenerate=False,
    transform=None,
):
    """
    Manages downloading, processing, partitioning, saving, and loading
    of windowed time series data for MULTIPLE tickers from a CSV list,
    returning a combined Dataset object for the specified split.

    Args:
        ticker_list_csv_path (str): Path to CSV file with ticker symbols (one per row, assumes first column).
        root (str): Root directory for storing raw and processed data.
        lookback_window (int): Input sequence length.
        prediction_horizon (int): Output sequence length.
        split (str): Which data split to return ('train', 'val', or 'test').
        target_column (str or list): Target column name(s) for prediction.
        feature_columns (list, optional): Feature column names. Defaults to OHLCV.
        download (bool): Whether to download raw data if missing.
        split_percentages (tuple): Fractions for train, val, test splits.
        force_regenerate (bool): If True, re-process data even if files exist.
        transform (callable, optional): Transformations for the Dataset object (applied if loading pre-saved).

    Returns:
        torch.utils.data.Dataset: A PyTorch Dataset instance (TensorDataset or TimeSeriesDataset)
                                   for the specified split, combining data from all tickers.
    """
    # --- Input Validation ---
    if not os.path.exists(ticker_list_csv_path):
        raise FileNotFoundError(f"Ticker list CSV not found: {ticker_list_csv_path}")
    if not (
        isinstance(split_percentages, (list, tuple))
        and len(split_percentages) == 3
        and abs(sum(split_percentages) - 1.0) < 1e-6
    ):
        raise ValueError(
            "split_percentages must be a list/tuple of 3 values summing to 1.0"
        )
    if split not in ["train", "val", "test"]:
        raise ValueError("split must be one of 'train', 'val', 'test'")

    # --- Read Ticker List ---
    try:
        ticker_df = pd.read_csv(ticker_list_csv_path)
        if ticker_df.empty:
            raise ValueError("Ticker list CSV is empty.")
        # Assume tickers are in the first column
        ticker_list = ticker_df.iloc[:, 0].astype(str).unique().tolist()
        print(f"Found tickers: {ticker_list}")
    except Exception as e:
        raise ValueError(
            f"Failed to read or parse ticker list CSV {ticker_list_csv_path}: {e}"
        )

    # --- Define Paths and Filenames ---
    raw_dir = os.path.join(root, "raw")
    processed_dir = os.path.join(root, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Create a unique identifier based on parameters including the list of tickers
    ticker_hash = hashlib.md5(str(sorted(ticker_list)).encode()).hexdigest()[
        :8
    ]  # Hash ticker list for filename
    target_str = (
        target_column
        if isinstance(target_column, str)
        else "-".join(sorted(target_column))
    )
    feature_str = (
        "default"
        if feature_columns is None
        else "-".join(sorted(f.replace(" ", "") for f in feature_columns))
    )

    base_filename = f"multi_{ticker_hash}_lb{lookback_window}_h{prediction_horizon}_tgt_{target_str}_feat_{feature_str}"
    train_file = os.path.join(processed_dir, f"{base_filename}_train.npz")
    val_file = os.path.join(processed_dir, f"{base_filename}_val.npz")
    test_file = os.path.join(processed_dir, f"{base_filename}_test.npz")
    processed_files = {"train": train_file, "val": val_file, "test": test_file}

    target_processed_file = processed_files[split]
    print(f"Target processed file for split '{split}': {target_processed_file}")

    # --- Check if Processed Data Exists ---
    if not force_regenerate and os.path.exists(target_processed_file):
        print(f"Processed file found for split '{split}'. Loading data.")
        try:
            # Use TimeSeriesDataset to load the specific split file

            loaded_data = np.load(target_processed_file)
            if "X" not in loaded_data or "y" not in loaded_data:
                raise ValueError(
                    f".npz file must contain arrays named 'X' and 'y'. Found: {list(loaded_data.keys())}"
                )
            X_data = torch.tensor(loaded_data["X"], dtype=torch.float32)
            y_data = torch.tensor(loaded_data["y"], dtype=torch.float32)
            print("Data loaded successfully.")

            return TensorDataset(X_data, y_data)
        except Exception as e:
            print(
                f"[Warning] Found processed file {target_processed_file} but failed to load: {e}. Will attempt regeneration."
            )

    # --- Generate Data (If necessary or forced) ---
    print(
        f"Processed file for split '{split}' not found or regeneration forced. Generating combined data..."
    )

    all_X_list = []
    all_y_list = []

    for ticker in ticker_list:
        print(f"\nProcessing ticker: {ticker}")
        # 1. Check/Download Raw Data for current ticker
        safe_ticker = ticker.replace(".", "_").replace("^", "")
        raw_csv_path = os.path.join(raw_dir, f"{safe_ticker}.csv")
        if not os.path.exists(raw_csv_path):
            if download:
                print(f"Raw file not found for {ticker}. Downloading...")
                raw_csv_path_result = getData(
                    ticker, output_folder=raw_dir, filename=f"{safe_ticker}.csv"
                )
                if raw_csv_path_result is None:
                    print(
                        f"[Warning] Failed to download raw data for {ticker}. Skipping this ticker."
                    )
                    continue  # Skip to next ticker
                raw_csv_path = raw_csv_path_result
            else:
                print(
                    f"[Warning] Raw data file not found for {ticker}: {raw_csv_path}. Skipping this ticker (download=False)."
                )
                continue  # Skip to next ticker

        # 2. Create Windows for current ticker
        X_ticker, y_ticker = create_sliding_windows(
            csv_path=raw_csv_path,
            lookback_window=lookback_window,
            prediction_horizon=prediction_horizon,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        if X_ticker is not None and y_ticker is not None:
            all_X_list.append(X_ticker)
            all_y_list.append(y_ticker)
        else:
            print(
                f"[Warning] Failed to create windows for {ticker}. Skipping this ticker."
            )

    # Check if any data was processed
    if not all_X_list:
        raise RuntimeError("Failed to process window data for any ticker in the list.")

    # 3. Concatenate data from all tickers
    print("\nConcatenating data from all processed tickers...")
    X_full = np.concatenate(all_X_list, axis=0)
    y_full = np.concatenate(all_y_list, axis=0)
    print(f"Combined dataset shape: X={X_full.shape}, y={y_full.shape}")
    # Note: Simple concatenation might mix timelines if stocks have different start/end dates.
    # Consider shuffling the combined data *before* partitioning if appropriate for your task,
    # OR partition each stock first then combine partitions if strict chronology per stock is needed.
    # For now, we proceed with partitioning the concatenated data chronologically.

    # 4. Partition Combined Data
    total_samples = X_full.shape[0]
    train_end_idx = int(total_samples * split_percentages[0])
    val_end_idx = train_end_idx + int(total_samples * split_percentages[1])

    X_train, y_train = X_full[:train_end_idx], y_full[:train_end_idx]
    X_val, y_val = X_full[train_end_idx:val_end_idx], y_full[train_end_idx:val_end_idx]
    X_test, y_test = X_full[val_end_idx:], y_full[val_end_idx:]

    partitions = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    # 5. Save Processed Partitions
    print("Saving processed partitions...")
    try:
        np.savez_compressed(train_file, X=X_train, y=y_train)
        np.savez_compressed(val_file, X=X_val, y=y_val)
        np.savez_compressed(test_file, X=X_test, y=y_test)
        print(f"Saved train data to {train_file}")
        print(f"Saved val data to {val_file}")
        print(f"Saved test data to {test_file}")
    except Exception as e:
        print(f"[Error] Failed to save processed data partitions: {e}")

    # 6. Return the requested split as a Dataset object
    X_selected, y_selected = partitions[split]
    print(f"Returning '{split}' split as a TensorDataset object.")
    # Using TensorDataset for simplicity after generating in memory
    return TensorDataset(
        torch.tensor(X_selected, dtype=torch.float32),
        torch.tensor(y_selected, dtype=torch.float32),
    )


if __name__ == "__main__":
    # test the get data function
    ticker = "AAPL"
    root = "./data/csv/"
    getData(ticker, output_folder=root, filename=f"{ticker}.csv")
    # test the createDataset function
