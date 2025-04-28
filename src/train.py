import torch
import torch.nn as nn
import torch.optim as optim
from data_pipeline import DataPipeline
from hidformer import Hidformer

def main():
    tickers = ['MSFT','AAPL','NVDA','TSM']
    dp = DataPipeline(
        tickers,
        start_date='2020-01-01',
        end_date='2023-12-31',
        seq_len=60,
        pred_len=10,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=32
    )
    dp.prepare_data()
    train_loader, val_loader, test_loader = dp.get_loaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Hidformer(
        input_dim=len(tickers),
        token_length=16,
        stride=8,
        time_blocks=4,
        freq_blocks=2,
        hidden_size=128,
        freq_k=64,
        out_dim=len(tickers)*10
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 20

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X).view(-1, 10, len(tickers))
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                pv = model(Xv).view(-1, 10, len(tickers))
                val_loss += criterion(pv, yv).item() * Xv.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} — Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    # Test
    test_loss = 0.0
    with torch.no_grad():
        for Xt, yt in test_loader:
            Xt, yt = Xt.to(device), yt.to(device)
            pt = model(Xt).view(-1, 10, len(tickers))
            test_loss += criterion(pt, yt).item() * Xt.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test MSE: {test_loss:.4f}")

if __name__=='__main__':
    main()
