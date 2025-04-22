import torch
import torch.nn as nn
import torch.optim as optim
from data_pipeline import DataPipeline

from hidformer import Hidformer


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=5,
        verbose=False,
        delta=0,
        path="./model/checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5 (as mentioned in Hidformer paper)
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                                   Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def main():
    # 2.1 — Configure & prepare data

    tickers = ["MSFT", "AAPL", "NVDA", "TSM"]
    seq_len = 60  # lookback window
    pred_len = 10  # forecast horizon
    batch_sz = 32
    dp = DataPipeline(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2023-12-31",
        seq_len=seq_len,
        pred_len=pred_len,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=batch_sz,
    )
    dp.prepare_data()
    train_loader, val_loader, test_loader = dp.get_loaders()
    print(
        f"Train samples: {len(dp.train_ds)}, Val: {len(dp.val_ds)}, Test: {len(dp.test_ds)}"
    )

    # 2.2 — Instantiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_dim = number of tickers = 4
    model = Hidformer(
        input_dim=len(tickers),
        token_length=16,
        stride=8,
        time_blocks=4,
        freq_blocks=2,
        hidden_size=128,
        freq_k=64,
        out_dim=len(tickers) * pred_len,  # predict pred_len steps for each ticker
    ).to(device)

    # 2.3 — Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 2.4 — Training loop
    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            # X: (B, seq_len, 4),  y: (B, pred_len, 4)
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            # pred: (B, 4*pred_len) — reshape to match y
            pred = pred.view(-1, pred_len, len(tickers))
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        train_loss = running_loss / len(dp.train_ds)

        # 2.5 — Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                pv = model(Xv).view(-1, pred_len, len(tickers))
                val_loss += criterion(pv, yv).item() * Xv.size(0)
        val_loss /= len(dp.val_ds)

        print(f"Epoch {epoch:02d} — Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    # 2.6 — Test set
    test_loss = 0.0
    with torch.no_grad():
        for Xt, yt in test_loader:
            Xt, yt = Xt.to(device), yt.to(device)
            pt = model(Xt).view(-1, pred_len, len(tickers))
            test_loss += criterion(pt, yt).item() * Xt.size(0)
    test_loss /= len(dp.test_ds)
    print(f"Test MSE: {test_loss:.4f}")


if __name__ == "__main__":
    main()
