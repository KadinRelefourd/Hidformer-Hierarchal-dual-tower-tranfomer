import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

def preprocess_prices(price_df):
    """
    Fill forward missing, drop any remaining NaNs, scale to [0,1].
    Returns numpy array (T, C) and fitted scaler.
    """
    df = price_df.fillna(method='ffill').dropna()
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df.values)
    return arr, scaler


def create_sequences(data_array, seq_len, pred_len):
    """
    Create sliding windows: X of shape (N, seq_len, C), y of shape (N, pred_len, C).
    """
    X, y = [], []
    n = len(data_array) - seq_len - pred_len + 1
    for i in range(n):
        X.append(data_array[i : i + seq_len])
        y.append(data_array[i + seq_len : i + seq_len + pred_len])
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    """
    Wraps numpy X, y into torch Dataset.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
