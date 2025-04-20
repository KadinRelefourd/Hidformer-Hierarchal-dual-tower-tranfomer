import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
import yfinance as yf
import pandas as pd
import os

#Blocks

class SRUppBlock(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(SRUppBlock, self).__init__()

class LinearAttentionBlock(nn.Module):
    """
    linear-attention block for the Frequency Tower token mixer.
    """
    def __init__(self, d_model, low_rank_k=32):
        super(LinearAttentionBlock, self).__init__()
        
        
#Towers
class TimeTowerBlock(nn.Module):
    """
    
    """
    def __init__(self, d_model, d_hidden):
        super(TimeTowerBlock, self).__init__()
        
class FrequencyTowerBlock(nn.Module):
    """
    
    """
    def __init__(self, d_model, low_rank_k=32):
        super(FrequencyTowerBlock, self).__init__()




#Hidformer

class Hidformer(nn.Module):
    """
    The main Hidformer model skeleton, with:
      - Token segmentation
      - Time tower
      - Frequency tower
      - Merge outputs & final decoder
    """
    def __init__(
        self,
        d_model=32,           # dimension for each segment embedding
        time_blocks=3,
        freq_blocks=2,
        d_hidden=32,          # hidden dimension inside SRU++ or MLP
        low_rank_k=32,        # dimension for linear attention projection
        segment_len=16,       # length of each time segment
        stride=8,             # stride for overlapping tokens
        out_len=1             # how many future steps to predict
    ):
        super(Hidformer, self).__init__()
        
        


# Token segmentation




# Training Loop
def train(model, train_loader, val_loader, num_epochs=10):
    """
    Train the Hidformer model.
    """
    



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
            auto_adjust=True # Get adjusted OHLC
        )

        # --- Handle Potential MultiIndex Columns ---
        # Although less common with auto_adjust=True, check just in case.
        if isinstance(ticker_data.columns, pd.MultiIndex):
            print(f"[Info] Detected multi-level columns for {ticker_symbol}. Flattening.")
            ticker_data.columns = ticker_data.columns.get_level_values(0)
            ticker_data = ticker_data.loc[:,~ticker_data.columns.duplicated(keep='first')]

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
        desired_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Select only the desired columns that ACTUALLY EXIST in the downloaded data
        available_columns = [col for col in desired_columns if col in ticker_data.columns]
        print(f"[Info] Columns available after download for {ticker_symbol}: {list(ticker_data.columns)}")
        print(f"[Info] Selecting columns for CSV: {available_columns}")

        if 'Date' not in available_columns:
             print(f"[Error] 'Date' column is unexpectedly missing for {ticker_symbol}. Skipping save.")
             return
        if not any(col in available_columns for col in ['Open', 'High', 'Low', 'Close']):
             print(f"[Error] No OHLC columns found for {ticker_symbol}. Skipping save.")
             return

        ticker_data_final = ticker_data[available_columns]

        # --- File and Folder Handling ---
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                print(f"Created output folder: {output_folder}")
            except OSError as e:
                print(f"[Error] Could not create output folder {output_folder}: {e}. Saving to current directory instead.")
                output_folder = "."

        if filename is None:
            # Default filename format: TICKER_adj_ohlcv.csv
            output_filename = f"{ticker_symbol.replace('.', '_')}.csv"
        else:
            # Use provided filename, ensuring it ends with .csv
            if not filename.lower().endswith('.csv'):
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

        print(f"[Success] Successfully downloaded and saved Adjusted OHLCV data for {ticker_symbol} to {output_path}")
        print(f"CSV Header Written: {','.join(ticker_data_final.columns)}")

    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"[Error] An unexpected error occurred while processing ticker {ticker_symbol}: {e}")



if __name__ == "__main__":
    getData("AAPL")
    # # Example usage
    # model = Hidformer()
    # print(model)
    
    # # Dummy data
    # x = torch.randn(32, 16, 32)  # (batch_size, seq_len, d_model)
    
    # # Forward pass
    # output = model(x)
    # print(output.shape)
