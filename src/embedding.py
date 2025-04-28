]import pandas as pd
import yfinance as yf

def download_data(tickers, start_date, end_date, interval='1d'):
    """
    Download adjusted-close prices for each ticker and concatenate into one DataFrame.
    """
    df_list = []
    for t in tickers:
        df_t = (
            yf.download(t, start=start_date, end=end_date, interval=interval, progress=False)
              .get("Adj Close")
              .rename(t)
        )
        df_list.append(df_t)
    price_df = pd.concat(df_list, axis=1)
    price_df.index.name = "Date"
    return price_df
