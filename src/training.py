import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_pipeline import DataPipeline
from hidformer import Hidformer

# Hyperparameter grid
time_blocks_list = [4, 5]
freq_blocks_list = [2, 3]
token_lengths = [16, 24]

# Data
tickers = ['MSFT','AAPL','NVDA','TSM']
seq_len, pred_len = 60, 10
batch_sz = 32
val_ratio, test_ratio = 0.1, 0.1

dp = DataPipeline(
    tickers, '2020-01-01', '2023-12-31',
    seq_len, pred_len, val_ratio, test_ratio, batch_sz
)
dp.prepare_data()
train_loader, val_loader, test_loader = dp.get_loaders()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = []
histories = {}

for tb in time_blocks_list:
    for fb in freq_blocks_list:
        for tl in token_lengths:
            # Instantiate model
            model = Hidformer(
                input_dim=len(tickers),
                token_length=tl,
                stride=8,
                time_blocks=tb,
                freq_blocks=fb,
                hidden_size=128,
                freq_k=64,
                out_dim=len(tickers)*pred_len
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            train_losses, val_losses = [], []
            epochs = 20

            for ep in range(1, epochs+1):
                model.train()
                run_loss = 0.0
                for X,y in train_loader:
                    X,y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    p = model(X).view(-1, pred_len, len(tickers))
                    loss = criterion(p, y)
                    loss.backward()
                    optimizer.step()
                    run_loss += loss.item() * X.size(0)
                train_losses.append(run_loss / len(train_loader.dataset))

                model.eval()
                vrun = 0.0
                with torch.no_grad():
                    for Xv,yv in val_loader:
                        Xv,yv = Xv.to(device), yv.to(device)
                        pv = model(Xv).view(-1, pred_len, len(tickers))
                        vrun += criterion(pv, yv).item() * Xv.size(0)
                val_losses.append(vrun / len(val_loader.dataset))

            # Test eval
            test_run = 0.0
            with torch.no_grad():
                for Xt, yt in test_loader:
                    Xt,yt = Xt.to(device), yt.to(device)
                    pt = model(Xt).view(-1, pred_len, len(tickers))
                    test_run += criterion(pt, yt).item() * Xt.size(0)
            test_loss = test_run / len(test_loader.dataset)

            results.append({'tb':tb,'fb':fb,'tl':tl,'test_mse':test_loss})
            histories[(tb,fb,tl)] = {'train':train_losses,'val':val_losses}

# Plot
for tl in token_lengths:
    plt.figure()
    for tb in time_blocks_list:
        for fb in freq_blocks_list:
            key = (tb,fb,tl)
            plt.plot(
                range(1, epochs+1),
                histories[key]['val'],
                label=f'TB{tb}-FB{fb}'
            )
    plt.title(f'Val Loss Curves (Token Length={tl})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
