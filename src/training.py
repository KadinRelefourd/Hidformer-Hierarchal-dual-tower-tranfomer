import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from preprocessing import createDataset
from hidformer import Hidformer

# casual comments like a 9th grader might write
# this script runs experiments and visualizes actual vs. predicted

def run_experiment(time_blocks, freq_blocks, token_length,
                   lookback_window, prediction_horizon,
                   feature_columns, target_column,
                   batch_size, learning_rate, epochs,
                   patience, data_csv, data_root,
                   device):
    # prepare data
    ds_train = createDataset(
        ticker_list_csv_path=data_csv, root=data_root,
        lookback_window=lookback_window, prediction_horizon=prediction_horizon,
        split="train", target_column=target_column,
        feature_columns=feature_columns, download=False,
        force_regenerate=False)
    ds_val = createDataset(
        ticker_list_csv_path=data_csv, root=data_root,
        lookback_window=lookback_window, prediction_horizon=prediction_horizon,
        split="val", target_column=target_column,
        feature_columns=feature_columns, download=False,
        force_regenerate=False)
    ds_test = createDataset(
        ticker_list_csv_path=data_csv, root=data_root,
        lookback_window=lookback_window, prediction_horizon=prediction_horizon,
        split="test", target_column=target_column,
        feature_columns=feature_columns, download=False,
        force_regenerate=False)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # set up model defaults
    d_model = 128; freq_k = 64; merge_mode = "linear"; merge_k = 2
    def build_model():
        return Hidformer(
            input_dim=len(feature_columns), pred_len=prediction_horizon,
            token_length=token_length, stride=token_length,
            num_time_blocks=time_blocks, num_freq_blocks=freq_blocks,
            d_model=d_model, freq_k=freq_k,
            dropout=0.2, merge_mode=merge_mode, merge_k=merge_k
        ).to(device)

    model = build_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop with early stopping
    best_val_loss = float('inf'); stop_counter = 0
    for epoch in range(1, epochs+1):
        model.train(); train_loss=0
        for X, y in loader_train:
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad(); pred=model(X)
            if y.shape[-1]!=pred.shape[-1]:
                idx=feature_columns.index(target_column)
                pred=pred[:,:,idx:idx+1]
            loss=criterion(pred,y); loss.backward(); optimizer.step()
            train_loss+=loss.item()
        train_loss/=len(loader_train)
        # validation
        model.eval(); val_loss=0
        with torch.no_grad():
            for Xv,yv in loader_val:
                Xv,yv=Xv.to(device), yv.to(device)
                pv=model(Xv)
                if yv.shape[-1]!=pv.shape[-1]:
                    idx=feature_columns.index(target_column)
                    pv=pv[:,:,idx:idx+1]
                val_loss+=criterion(pv,yv).item()
        val_loss/=len(loader_val)
        if val_loss<best_val_loss:
            best_val_loss=val_loss; stop_counter=0; torch.save(model.state_dict(), 'best_model.pth')
        else:
            stop_counter+=1
        if stop_counter>=patience: break

    # load best and test
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval(); test_loss=0
    # for visualization, collect one batch
    X_sample, y_sample = next(iter(loader_test))
    Xs, ys = X_sample.to(device), y_sample.to(device)
    with torch.no_grad():
        preds = model(Xs)
        if ys.shape[-1]!=preds.shape[-1]:
            idx = feature_columns.index(target_column)
            preds = preds[:,:,idx:idx+1]
    # compute test loss
    test_loss = criterion(preds, ys).item()

    # return metrics and series
    return dict(
        time_blocks=time_blocks, freq_blocks=freq_blocks,
        token_length=token_length, best_val_loss=best_val_loss,
        test_loss=test_loss,
        true_series=ys.cpu().numpy(), pred_series=preds.cpu().numpy()
    )


def plot_actual_vs_pred(true_series, pred_series, config_name):
    # true_series, pred_series shapes: (B, H, 1)
    plt.figure(figsize=(10,4))
    # plot first sample
    ts = true_series[0,:,0]; ps = pred_series[0,:,0]
    plt.plot(ts, label='Actual')
    plt.plot(ps, label='Predicted')
    plt.title(f'Actual vs Predicted ({config_name})')
    plt.xlabel('Time step'); plt.ylabel('Value')
    plt.legend(); plt.tight_layout()
    plt.show()


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_CSV = os.path.join(BASE_DIR, 'src', 'tickers.csv')
    DATA_ROOT = os.path.join(BASE_DIR, 'data')
    LOOKBACK=60; PRED=10
    FEATURES=['Open','High','Low','Close','Volume']; TARGET='Close'
    BATCH=32; LR=1e-4; EPOCHS=30; PATIENCE=5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # only keep sweep CSV, but visualize actual vs predicted for token lengths
    token_lengths = [8,16,32]
    results=[]
    for tl in token_lengths:
        res = run_experiment(
            time_blocks=4, freq_blocks=2, token_length=tl,
            lookback_window=LOOKBACK, prediction_horizon=PRED,
            feature_columns=FEATURES, target_column=TARGET,
            batch_size=BATCH, learning_rate=LR,
            epochs=EPOCHS, patience=PATIENCE,
            data_csv=DATA_CSV, data_root=DATA_ROOT,
            device=device
        )
        results.append({k:v for k,v in res.items() if k not in ['true_series','pred_series']})
        config_name = f'tl={tl}'
        plot_actual_vs_pred(res['true_series'], res['pred_series'], config_name)

    # save summary
    df = pd.DataFrame(results)
    df.to_csv('sweep_results.csv', index=False)
    print('Completed. Sweep results and plots done.')

if __name__=='__main__': main()
