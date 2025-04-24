import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from data_pipeline import DataPipeline
from hidformer import Hidformer
import ace_tools as tools

# Hyperparameter grid
time_freq_pairs = [(4, 2), (4, 3), (5, 2), (5, 3)]
token_lengths = [16, 24, 32]

# Configuration
tickers = ['MSFT', 'AAPL', 'NVDA', 'TSM']
seq_len = 60
pred_len = 10
batch_sz = 32
val_ratio = 0.1
test_ratio = 0.1
epochs = 20
learning_rate = 1e-4

# Prepare data once
dp = DataPipeline(
    tickers=tickers,
    start_date='2020-01-01',
    end_date='2023-12-31',
    seq_len=seq_len,
    pred_len=pred_len,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
    batch_size=batch_sz
)
dp.prepare_data()
train_loader, val_loader, test_loader = dp.get_loaders()

results = []
histories = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for time_blocks, freq_blocks in time_freq_pairs:
    for token_length in token_lengths:
        # Instantiate model
        model = Hidformer(
            input_dim=len(tickers),
            token_length=token_length,
            stride=8,
            time_blocks=time_blocks,
            freq_blocks=freq_blocks,
            hidden_size=128,
            freq_k=64,
            out_dim=len(tickers) * pred_len
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(X).view(-1, pred_len, len(tickers))
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X.size(0)
            train_losses.append(running_loss / len(dp.train_ds))

            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device), yv.to(device)
                    pv = model(Xv).view(-1, pred_len, len(tickers))
                    val_running += criterion(pv, yv).item() * Xv.size(0)
            val_losses.append(val_running / len(dp.val_ds))

        # Test evaluation
        test_running = 0.0
        with torch.no_grad():
            for Xt, yt in test_loader:
                Xt, yt = Xt.to(device), yt.to(device)
                pt = model(Xt).view(-1, pred_len, len(tickers))
                test_running += criterion(pt, yt).item() * Xt.size(0)
        test_mse = test_running / len(dp.test_ds)

        # Record results
        results.append({
            'time_blocks': time_blocks,
            'freq_blocks': freq_blocks,
            'token_length': token_length,
            'test_mse': test_mse
        })
        histories[(time_blocks, freq_blocks, token_length)] = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }

# Create results DataFrame and display it
df_results = pd.DataFrame(results)
tools.display_dataframe_to_user("Model Comparison Results", df_results)

# Plot validation loss curves for each token length
for token_length in token_lengths:
    plt.figure()
    for time_blocks, freq_blocks in time_freq_pairs:
        key = (time_blocks, freq_blocks, token_length)
        val_losses = histories[key]['val_losses']
        plt.plot(range(1, epochs + 1), val_losses, label=f'TB{time_blocks}-FB{freq_blocks}')
    plt.title(f'Validation Loss Curves (Token Length={token_length})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
