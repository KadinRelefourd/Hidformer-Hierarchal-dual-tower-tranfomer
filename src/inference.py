import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
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

# --- Configuration ---
# These can be overridden by command-line arguments
DEFAULT_MODEL_PATH = "./model/test_train.pt"  # Path to the trained model
DEFAULT_TICKER = "AAPL"  # Ticker symbol for inference
DEFAULT_START_DATE = "2023-01-01"  # Start date for inference data
DEFAULT_END_DATE = "2024-01-01"  # End date for inference data

# --- Model & Data Parameters (MUST match the training configuration of the loaded model) ---
LOOKBACK_WINDOW = 128  # Lookback window (Hidformer input length T_in)
PREDICTION_HORIZON = 10  # Forecast horizon (Hidformer pred_len H)
FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]  # Features used during training
TARGET_COLUMN = "Close"  # Target variable predicted

# Model Hyperparameters (MUST match the trained model)
TOKEN_LENGTH = 16
STRIDE = 8
NUM_TIME_BLOCKS = 4
NUM_FREQ_BLOCKS = 2
D_MODEL = 128
FREQ_K = 64
DROPOUT = 0.2  # Dropout is inactive during model.eval()
MERGE_MODE = "linear"
MERGE_K = 2
INPUT_DIM = len(FEATURE_COLUMNS)  # Should match number of features

# --- Trading Strategy Parameters ---
SIGNAL_HORIZON = (
    1  # Use prediction for day t+SIGNAL_HORIZON as signal (e.g., 1 for next day)
)
BUY_THRESHOLD = 0.01  # Buy if predicted price > current_price * (1 + threshold)
SELL_THRESHOLD = 0.01  # Sell if predicted price < current_price * (1 - threshold)

# --- Helper Functions ---


def download_inference_data(ticker, start, end):
    """Downloads historical stock data for the specified ticker and date range."""
    print(f"--- Downloading inference data for: {ticker} ({start} to {end}) ---")
    try:
        # Use group_by='ticker' which often prevents MultiIndex when downloading single ticker
        # but we will handle MultiIndex explicitly just in case.
        ticker_data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,  # Get adjusted OHLC, Volume might not be adjusted
            group_by="ticker",  # Helps prevent MultiIndex for single ticker
        )

        if ticker_data is None or ticker_data.empty:
            print(f"[Warning] No data found for {ticker} in the specified range.")
            return None

        # --- Handle Potential MultiIndex Columns ---
        if isinstance(ticker_data.columns, pd.MultiIndex):
            print("[Info] Detected multi-level columns. Flattening...")
            ticker_data.columns = ticker_data.columns.get_level_values(
                1
            )  # Get 'Price' level
            ticker_data = ticker_data.loc[
                :, ~ticker_data.columns.duplicated(keep="first")
            ]
            ticker_data.columns.name = None  # Remove the index name
            print(f"[Info] Columns after flattening: {list(ticker_data.columns)}")

        # Ensure required columns exist, handle potential missing 'Volume'
        if "Volume" in FEATURE_COLUMNS and "Volume" not in ticker_data.columns:
            print(
                "[Warning] 'Volume' column missing after download, filling with zeros."
            )
            ticker_data["Volume"] = 0

        # Select only the features the model was trained on
        columns_to_select = [
            col for col in FEATURE_COLUMNS if col in ticker_data.columns
        ]
        if not columns_to_select or len(columns_to_select) != len(FEATURE_COLUMNS):
            missing_in_download = set(FEATURE_COLUMNS) - set(ticker_data.columns)
            print(
                f"[Error] Not all expected feature columns ({FEATURE_COLUMNS}) were found in the downloaded data."
            )
            if missing_in_download:
                print(f"Missing columns: {missing_in_download}")
            print(f"Available columns: {list(ticker_data.columns)}")
            return None

        ticker_data_selected = ticker_data[
            FEATURE_COLUMNS
        ]  # Select in the defined order

        print(
            f"Downloaded and selected {len(ticker_data_selected)} rows. Columns: {list(ticker_data_selected.columns)}"
        )
        return ticker_data_selected  # Keep Date as index

    except Exception as e:
        print(f"[Error] Failed to download or process data for {ticker}: {e}")
        return None


def preprocess_for_inference(df, feature_cols, lookback):
    """
    Scales features and creates overlapping sequences for inference.

    Args:
        df (pd.DataFrame): DataFrame with features and DatetimeIndex.
        feature_cols (list): List of columns to use as features (must match df columns).
        lookback (int): Sequence length for the model.

    Returns:
        tuple: (torch.Tensor, MinMaxScaler, pd.DatetimeIndex, pd.DatetimeIndex)
               - Tensor of input sequences (num_sequences, lookback, num_features).
               - Fitted scaler object.
               - DatetimeIndex of the end date for each sequence.
               - DatetimeIndex of all dates starting from the first possible prediction point.
    """
    print("Preprocessing data: Scaling features and creating sequences...")
    if not all(col in df.columns for col in feature_cols):
        print(
            f"[Error] DataFrame is missing one or more feature columns expected: {feature_cols}"
        )
        print(f"DataFrame columns: {list(df.columns)}")
        return None, None, None, None

    # --- Scaling ---
    # Fit scaler ONLY on the columns defined in feature_cols, in that specific order
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=feature_cols)
    print(
        f"Features scaled using MinMaxScaler (fitted on inference data): {feature_cols}"
    )

    # --- Create Sequences ---
    X = []
    sequence_end_dates = []  # Store corresponding end dates for sequences

    if len(scaled_df) < lookback:
        print(
            f"Error: Data length ({len(scaled_df)}) is less than lookback window ({lookback}). Cannot create sequences."
        )
        return None, None, None, None

    for i in range(len(scaled_df) - lookback + 1):
        sequence = scaled_df.iloc[i : i + lookback].values  # Use scaled data
        X.append(sequence)
        sequence_end_dates.append(scaled_df.index[i + lookback - 1])

    if not X:
        print("Error: No sequences were created.")
        return None, None, None, None

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    sequence_end_dates_idx = pd.DatetimeIndex(sequence_end_dates)

    # Determine the full date range relevant for plotting actuals against predictions
    first_prediction_date = sequence_end_dates_idx[0] + pd.Timedelta(days=1)
    try:
        plot_start_index = df.index.get_loc(first_prediction_date)
    except KeyError:
        available_dates_after = df.index[df.index > sequence_end_dates_idx[0]]
        if not available_dates_after.empty:
            first_prediction_date = available_dates_after[0]
            plot_start_index = df.index.get_loc(first_prediction_date)
            print(f"Adjusted first prediction plot date to {first_prediction_date}")
        else:
            print("Error: Cannot find valid dates after the first sequence end date.")
            return None, None, None, None

    all_dates_for_plotting = df.index[plot_start_index:]

    print(f"Created {X_tensor.shape[0]} sequences.")
    # Return the scaler along with other data
    return X_tensor, scaler, sequence_end_dates_idx, all_dates_for_plotting


def run_inference(model, data_tensor, scaler, device, batch_size=64):
    """
    Runs the model in evaluation mode to get predictions and denormalizes them.

    Args:
        model (torch.nn.Module): The loaded Hidformer model.
        data_tensor (torch.Tensor): Input sequences (num_sequences, lookback, features).
        scaler (MinMaxScaler): The fitted scaler object used for preprocessing.
        device (torch.device): CPU or CUDA device.
        batch_size (int): Batch size for inference to manage memory.

    Returns:
        np.ndarray: Explicitly denormalized predictions (num_sequences, pred_horizon, features).
    """
    print(f"Running inference on {data_tensor.shape[0]} sequences...")
    model.eval()  # Set model to evaluation mode
    raw_predictions = []
    with torch.no_grad():
        num_sequences = data_tensor.shape[0]
        for i in range(0, num_sequences, batch_size):
            batch_X = data_tensor[i : min(i + batch_size, num_sequences)].to(device)
            # Get model output - ASSUME this is SCALED output, ignore internal RevIN for now
            batch_pred = model(batch_X)
            raw_predictions.append(batch_pred.cpu().numpy())

    raw_predictions = np.concatenate(raw_predictions, axis=0)
    print("Raw inference complete (model output obtained).")
    print(f"Shape of raw_predictions: {raw_predictions.shape}")

    # --- Explicit Denormalization ---
    # Reshape for scaler: (num_sequences * pred_horizon, num_features)
    num_sequences, pred_horizon, num_features = raw_predictions.shape
    predictions_reshaped = raw_predictions.reshape(-1, num_features)

    # Check if scaler expects the same number of features
    if predictions_reshaped.shape[1] != scaler.n_features_in_:
        print(
            f"[Error] Mismatch between prediction features ({predictions_reshaped.shape[1]}) and scaler features ({scaler.n_features_in_}). Cannot denormalize."
        )
        # As a fallback, try denormalizing only the target column if shapes mismatch drastically
        # This assumes the target column index is consistent.
        target_col_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
        if target_col_idx < predictions_reshaped.shape[1]:
            print(
                "[Warning] Attempting fallback: Denormalizing only the target column."
            )
            # Create a dummy array matching scaler's expected features
            dummy_array = np.zeros(
                (predictions_reshaped.shape[0], scaler.n_features_in_)
            )
            # Place the predicted target column into the dummy array at the correct index
            dummy_array[:, target_col_idx] = predictions_reshaped[:, target_col_idx]
            # Inverse transform the dummy array
            denorm_dummy = scaler.inverse_transform(dummy_array)
            # Extract the denormalized target column
            denormalized_predictions_reshaped = denorm_dummy[
                :, target_col_idx : target_col_idx + 1
            ]  # Keep as 2D
            # We need to reconstruct the full prediction array shape, filling others with NaN or scaled values
            # This is complex and error-prone. Let's raise error for now.
            raise ValueError(
                "Feature mismatch during denormalization is too complex to handle automatically."
            )
        else:
            raise ValueError(
                f"Target column index {target_col_idx} out of bounds for prediction shape {predictions_reshaped.shape}"
            )

    print(
        f"Denormalizing {predictions_reshaped.shape[0]} samples with {scaler.n_features_in_} features..."
    )
    denormalized_predictions_reshaped = scaler.inverse_transform(predictions_reshaped)

    # Reshape back to original prediction shape: (num_sequences, pred_horizon, num_features)
    denormalized_predictions = denormalized_predictions_reshaped.reshape(
        num_sequences, pred_horizon, num_features
    )
    print("Denormalization complete.")

    # --- Debug: Print prediction stats AFTER denormalization ---
    print(f"Shape of denormalized_predictions: {denormalized_predictions.shape}")
    print(f"Min denormalized prediction value: {np.min(denormalized_predictions)}")
    print(f"Max denormalized prediction value: {np.max(denormalized_predictions)}")
    print(f"Mean denormalized prediction value: {np.mean(denormalized_predictions)}")
    target_col_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
    print(
        f"First few denormalized predictions (target column '{TARGET_COLUMN}'):\n",
        denormalized_predictions[:5, :, target_col_idx],
    )
    # --- End Debug ---

    return denormalized_predictions


def visualize_forecast(
    actual_data,
    sequence_end_dates,
    predicted_data,
    ticker,
    target_col_name,
    target_col_idx,
    horizon=1,
):
    """
    Plots actual vs predicted prices for a specific forecast horizon.

    Args:
        actual_data (pd.Series): Series of actual prices with DatetimeIndex.
        sequence_end_dates (pd.DatetimeIndex): Dates corresponding to the end of each input sequence.
        predicted_data (np.ndarray): Full prediction array (num_sequences, pred_horizon, features).
                                     *Assumed to be denormalized*.
        ticker (str): Stock ticker symbol.
        target_col_name (str): Name of the target column (e.g., 'Close').
        target_col_idx (int): Index of the target column in the prediction array's last dimension.
        horizon (int): The specific forecast horizon step to plot (e.g., 1 for 1-day ahead).
    """
    print(f"Visualizing forecast for horizon H={horizon}...")
    plt.style.use("seaborn-v0_8-darkgrid")  # Use a pleasant style
    plt.figure(figsize=(14, 7))

    # Extract predictions for the specific horizon (already denormalized)
    preds_h = predicted_data[:, horizon - 1, target_col_idx]

    # Create prediction dates: prediction `i` made using data up to `sequence_end_dates[i]`
    # forecasts the price for `sequence_end_dates[i] + horizon` days.
    pred_dates = []
    valid_preds_h = []
    original_dates = actual_data.index  # Get all valid trading dates from actuals

    for i, end_date in enumerate(sequence_end_dates):
        end_date_loc = original_dates.searchsorted(end_date)
        if (
            end_date_loc < len(original_dates)
            and original_dates[end_date_loc] == end_date
        ):
            target_date_idx = end_date_loc + horizon
        else:
            target_date_idx = end_date_loc + horizon - 1

        if target_date_idx < len(original_dates):
            actual_forecast_date = original_dates[target_date_idx]
            pred_dates.append(actual_forecast_date)
            valid_preds_h.append(preds_h[i])

    pred_series = pd.Series(valid_preds_h, index=pd.DatetimeIndex(pred_dates))

    # --- Debug: Print forecast series ---
    print(f"Forecast plot: Actual data head:\n{actual_data.head()}")
    print(f"Forecast plot: Actual data tail:\n{actual_data.tail()}")
    if not pred_series.empty:
        print(
            f"Forecast plot: Predicted series head (DENORMALIZED):\n{pred_series.head()}"
        )
        print(
            f"Forecast plot: Predicted series tail (DENORMALIZED):\n{pred_series.tail()}"
        )
    else:
        print("Forecast plot: Predicted series is empty.")
    # --- End Debug ---

    # Plot actual prices
    plt.plot(
        actual_data.index,
        actual_data,
        label=f"Actual {target_col_name}",
        color="royalblue",
        linewidth=2,
    )

    # Plot predictions (should now be on the correct scale)
    if not pred_series.empty:
        plt.plot(
            pred_series.index,
            pred_series,
            label=f"Predicted {target_col_name} (H={horizon})",
            color="darkorange",
            linestyle="--",
            linewidth=2,
        )
    else:
        print(f"Warning: No valid predictions to plot for horizon {horizon}.")

    plt.title(f"{ticker} Stock Price Forecast (Horizon={horizon} days)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def run_trading_strategy(
    full_actual_prices,
    sequence_end_dates,
    predicted_prices,
    target_col_idx,
    signal_horizon,
    buy_thresh,
    sell_thresh,
):
    """
    Implements and simulates a simple threshold-based trading strategy.

    Args:
        full_actual_prices (pd.Series): Series of actual prices covering the entire period, with DatetimeIndex.
        sequence_end_dates (pd.DatetimeIndex): Dates corresponding to the end of each input sequence.
        predicted_prices (np.ndarray): Full prediction array (num_sequences, pred_horizon, features).
                                     *Assumed to be denormalized*.
        target_col_idx (int): Index of the target column in predictions.
        signal_horizon (int): Forecast horizon used for generating signals.
        buy_thresh (float): Relative threshold for buy signal.
        sell_thresh (float): Relative threshold for sell signal.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame)
               - signals_df: DataFrame with signals and actions taken based on prediction date.
               - positions_df: DataFrame indicating position (0 or 1) held on each trading day.
    """
    print(f"Running trading strategy using H={signal_horizon} predictions...")
    # Align predictions with the date the decision is made (end of lookback window)
    # Predictions are now denormalized
    preds_signal_h = predicted_prices[:, signal_horizon - 1, target_col_idx]
    signals = pd.DataFrame(index=sequence_end_dates)
    signals["Predicted Price (t+h)"] = preds_signal_h

    # Get the actual price on the day the decision is made
    signals["Actual Price (t)"] = full_actual_prices.reindex(sequence_end_dates)

    signals.dropna(subset=["Actual Price (t)"], inplace=True)

    # --- Generate Signals ---
    signals["Signal"] = 0  # 0: Hold, 1: Buy, -1: Sell
    signals["Buy Threshold Price"] = signals["Actual Price (t)"] * (1 + buy_thresh)
    signals["Sell Threshold Price"] = signals["Actual Price (t)"] * (1 - sell_thresh)

    # Compare denormalized prediction with calculated thresholds
    buy_condition = signals["Predicted Price (t+h)"] > signals["Buy Threshold Price"]
    sell_condition = signals["Predicted Price (t+h)"] < signals["Sell Threshold Price"]
    signals.loc[buy_condition, "Signal"] = 1
    signals.loc[sell_condition, "Signal"] = -1

    # --- Debug: Print signal generation info ---
    print("\n--- Signal Generation Debug ---")
    print(f"Buy Threshold: {buy_thresh}, Sell Threshold: {sell_thresh}")
    print("Signals DataFrame head (using DENORMALIZED predictions):")
    print(
        signals[
            [
                "Actual Price (t)",
                "Predicted Price (t+h)",
                "Buy Threshold Price",
                "Sell Threshold Price",
                "Signal",
            ]
        ].head(10)
    )
    print("\nSignal Counts:")
    print(signals["Signal"].value_counts())
    print("--- End Signal Generation Debug ---\n")
    # --- End Debug ---

    # --- Simulate Positions ---
    positions = pd.DataFrame(index=full_actual_prices.index).fillna(0.0)
    positions["Position"] = 0
    current_position = 0
    signals["Action"] = "Hold"

    for date in signals.index:
        if date not in signals.index:
            continue  # Skip if date was dropped
        signal = signals.loc[date, "Signal"]
        trade_date = None
        available_dates_after = full_actual_prices.index[
            full_actual_prices.index > date
        ]
        if not available_dates_after.empty:
            trade_date = available_dates_after[0]
        else:
            continue

        if trade_date is not None and trade_date in positions.index:
            if signal == 1 and current_position == 0:
                positions.loc[trade_date:, "Position"] = 1
                current_position = 1
                signals.loc[date, "Action"] = "Buy"
            elif signal == -1 and current_position == 1:
                positions.loc[trade_date:, "Position"] = 0
                current_position = 0
                signals.loc[date, "Action"] = "Sell"

    print("Trading signals and positions generated.")
    return signals, positions


def visualize_strategy(actual_prices, signals, positions, ticker):
    """
    Visualizes the trading strategy signals and cumulative returns.

    Args:
        actual_prices (pd.Series): Series of actual prices with DatetimeIndex.
        signals (pd.DataFrame): DataFrame containing 'Action' column and indexed by decision date.
        positions (pd.DataFrame): DataFrame containing 'Position' column indexed by trading date.
        ticker (str): Stock ticker symbol.
    """
    print("Visualizing trading strategy...")
    plt.style.use("seaborn-v0_8-darkgrid")

    # --- Plot Price with Buy/Sell Signals ---
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(
        actual_prices.index,
        actual_prices,
        label="Actual Close",
        color="royalblue",
        linewidth=1.5,
        alpha=0.9,
    )
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Price", fontsize=12, color="royalblue")
    ax1.tick_params(axis="y", labelcolor="royalblue")
    ax1.grid(True, which="major", linestyle="--", linewidth=0.5)

    buy_action_dates = signals[signals["Action"] == "Buy"].index
    sell_action_dates = signals[signals["Action"] == "Sell"].index
    original_dates = actual_prices.index
    buy_trade_dates, sell_trade_dates = [], []

    for d in buy_action_dates:
        next_day_idx = original_dates.searchsorted(d)
        if next_day_idx < len(original_dates):
            if original_dates[next_day_idx] == d:
                next_day_idx += 1
            if next_day_idx < len(original_dates):
                buy_trade_dates.append(original_dates[next_day_idx])
    for d in sell_action_dates:
        next_day_idx = original_dates.searchsorted(d)
        if next_day_idx < len(original_dates):
            if original_dates[next_day_idx] == d:
                next_day_idx += 1
            if next_day_idx < len(original_dates):
                sell_trade_dates.append(original_dates[next_day_idx])

    buy_trade_dates = pd.DatetimeIndex(buy_trade_dates).unique()
    sell_trade_dates = pd.DatetimeIndex(sell_trade_dates).unique()

    valid_buy_dates = buy_trade_dates.intersection(actual_prices.index)
    if not valid_buy_dates.empty:
        ax1.plot(
            valid_buy_dates,
            actual_prices.loc[valid_buy_dates],
            "^",
            markersize=10,
            color="limegreen",
            markeredgewidth=1.5,
            markerfacecolor="none",
            label="Buy Execution",
        )
    else:
        print("Strategy plot: No valid buy execution dates found to plot.")
    valid_sell_dates = sell_trade_dates.intersection(actual_prices.index)
    if not valid_sell_dates.empty:
        ax1.plot(
            valid_sell_dates,
            actual_prices.loc[valid_sell_dates],
            "v",
            markersize=10,
            color="red",
            markeredgewidth=1.5,
            markerfacecolor="none",
            label="Sell Execution",
        )
    else:
        print("Strategy plot: No valid sell execution dates found to plot.")

    ax1.set_title(f"{ticker} Trading Strategy Backtest: Signals", fontsize=16)
    ax1.legend(loc="upper left", fontsize=10)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # --- Plot Cumulative Returns ---
    returns = actual_prices.pct_change().fillna(0)
    strategy_returns = positions["Position"].shift(1) * returns
    strategy_returns = strategy_returns.fillna(0)
    cumulative_strategy_returns = (1 + strategy_returns).cumprod()
    cumulative_buy_hold_returns = (1 + returns).cumprod()

    # --- Debug: Print returns info ---
    print("\n--- Returns Calculation Debug ---")
    print("Positions head:")
    print(positions.head())
    print("\nPositions counts:")
    print(positions["Position"].value_counts())
    print("\nStock returns head:")
    print(returns.head())
    print("\nStrategy returns head:")
    print(strategy_returns.head())
    print("\nCumulative Strategy returns head:")
    print(cumulative_strategy_returns.head())
    print("\nCumulative Strategy returns tail:")
    print(cumulative_strategy_returns.tail())
    print("\nCumulative Buy & Hold returns head:")
    print(cumulative_buy_hold_returns.head())
    print("\nCumulative Buy & Hold returns tail:")
    print(cumulative_buy_hold_returns.tail())
    print("--- End Returns Calculation Debug ---\n")
    # --- End Debug ---

    plt.figure(figsize=(14, 7))
    plt.plot(
        cumulative_strategy_returns.index,
        cumulative_strategy_returns,
        label="Strategy Cumulative Returns",
        color="purple",
        linewidth=2,
    )
    plt.plot(
        cumulative_buy_hold_returns.index,
        cumulative_buy_hold_returns,
        label="Buy & Hold Cumulative Returns",
        color="grey",
        linestyle="--",
        linewidth=1.5,
    )
    plt.title(f"{ticker} Strategy vs. Buy & Hold Cumulative Returns", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Returns (1 = Initial Investment)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)
    # Adjust y-axis limits if returns are very close to 1
    min_ret = cumulative_strategy_returns.min()
    max_ret = cumulative_strategy_returns.max()
    if max_ret < 1.1 and min_ret > 0.9:
        plt.ylim(min_ret - 0.05, max_ret + 0.05)  # Add some padding
    elif max_ret == 1.0 and min_ret == 1.0:  # Handle completely flat case
        plt.ylim(0.95, 1.05)
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference and backtest trading strategy using a trained Hidformer model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model (.pt file)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=DEFAULT_TICKER,
        help="Stock ticker symbol for inference",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Start date for inference data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=DEFAULT_END_DATE,
        help="End date for inference data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=LOOKBACK_WINDOW,
        help="Lookback window size (must match training)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=PREDICTION_HORIZON,
        help="Prediction horizon (must match training)",
    )
    parser.add_argument(
        "--signal_h",
        type=int,
        default=SIGNAL_HORIZON,
        help="Prediction horizon step to use for trading signal",
    )
    parser.add_argument(
        "--buy_thresh",
        type=float,
        default=BUY_THRESHOLD,
        help="Buy threshold (relative)",
    )
    parser.add_argument(
        "--sell_thresh",
        type=float,
        default=SELL_THRESHOLD,
        help="Sell threshold (relative)",
    )

    args = parser.parse_args()

    # --- Validate Configuration ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit()
    if args.signal_h < 1 or args.signal_h > args.horizon:
        print(
            f"Error: Signal horizon ({args.signal_h}) must be between 1 and prediction horizon ({args.horizon})."
        )
        exit()
    LOOKBACK_WINDOW = args.lookback
    PREDICTION_HORIZON = args.horizon
    SIGNAL_HORIZON = args.signal_h
    BUY_THRESHOLD = args.buy_thresh
    SELL_THRESHOLD = args.sell_thresh

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    try:
        model = Hidformer(
            input_dim=INPUT_DIM,
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
        model.load_state_dict(
            torch.load(args.model_path, map_location=device), strict=False
        )
        model.eval()
        print(
            "Model loaded successfully (using strict=False). Check warnings above for missing/unexpected keys."
        )
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print(
            "Ensure the model architecture parameters in this script match the saved model."
        )
        exit()

    # 3. Load and Preprocess Inference Data
    df_inference = download_inference_data(args.ticker, args.start_date, args.end_date)
    if df_inference is None or len(df_inference) < LOOKBACK_WINDOW:
        print(
            f"Exiting: Not enough data for inference (need at least {LOOKBACK_WINDOW} days). Found {len(df_inference) if df_inference is not None else 'None'} days."
        )
        exit()

    # Pass FEATURE_COLUMNS explicitly to ensure order and content match scaler fitting
    X_inference, scaler, sequence_end_dates, all_dates_for_plotting = (
        preprocess_for_inference(df_inference, FEATURE_COLUMNS, LOOKBACK_WINDOW)
    )
    if X_inference is None:
        print("Exiting: Failed to preprocess data or create inference sequences.")
        exit()

    # 4. Run Inference and Denormalize
    # Pass the scaler to run_inference for explicit denormalization
    predictions_denorm = run_inference(model, X_inference, scaler, device)
    if predictions_denorm is None:
        print("Exiting: Failed to run inference or denormalize predictions.")
        exit()

    # 5. Prepare Data for Visualization and Strategy
    try:
        target_col_index = FEATURE_COLUMNS.index(TARGET_COLUMN)
    except ValueError:
        print(
            f"Error: Target column '{TARGET_COLUMN}' not found in FEATURE_COLUMNS: {FEATURE_COLUMNS}"
        )
        exit()

    actual_target_prices = df_inference[TARGET_COLUMN]  # Full series for alignment

    if len(sequence_end_dates) != predictions_denorm.shape[0]:
        print(
            f"Critical Warning: Mismatch between sequence end dates ({len(sequence_end_dates)}) and predictions ({predictions_denorm.shape[0]}). Results may be unreliable."
        )
        min_len = min(len(sequence_end_dates), predictions_denorm.shape[0])
        sequence_end_dates = sequence_end_dates[:min_len]
        predictions_denorm = predictions_denorm[:min_len]
        print(f"Attempted alignment to {min_len} predictions.")

    # 6. Visualize Forecast
    visualize_forecast(
        actual_target_prices,
        sequence_end_dates,
        predictions_denorm,
        args.ticker,
        TARGET_COLUMN,
        target_col_index,
        horizon=SIGNAL_HORIZON,
    )

    # 7. Run Trading Strategy
    signals_df, positions_df = run_trading_strategy(
        actual_target_prices,
        sequence_end_dates,
        predictions_denorm,
        target_col_index,
        SIGNAL_HORIZON,
        BUY_THRESHOLD,
        SELL_THRESHOLD,
    )
    print("\n--- Trading Signals Summary ---")
    print(signals_df["Action"].value_counts())

    # 8. Visualize Strategy Results
    visualize_strategy(actual_target_prices, signals_df, positions_df, args.ticker)

    print("\n--- Inference and Backtesting Complete ---")
