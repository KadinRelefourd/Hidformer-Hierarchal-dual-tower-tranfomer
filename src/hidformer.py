import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
import yfinance as yf
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
from preprocessing import createDataset

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
 

if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    #getData("AAPL")
    createDataset(
        ticker_list_csv_path="./src/tickers.csv",
        root="./data/",
        lookback_window=16,
        prediction_horizon=7,
        split='train',
        target_column='Close',
        feature_columns=None,
        download=True,
        split_percentages=(0.7, 0.15, 0.15),
        force_regenerate=False
    )
    # # Example usage
    # model = Hidformer()
    # print(model)
    
    # # Dummy data
    # x = torch.randn(32, 16, 32)  # (batch_size, seq_len, d_model)
    
    # # Forward pass
    # output = model(x)
    # print(output.shape)
