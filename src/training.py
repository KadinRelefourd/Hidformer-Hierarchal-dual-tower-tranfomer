import time
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
# this script runs experiments for different Hidformer parameter combos
def run_experiment(time_blocks, freq_blocks, token_length,
                   lookback_window, prediction_horizon,
                   feature_columns, target_column,
                   batch_size, learning_rate, epochs,
                   patience, data_csv, data_root,
                   device):
    # set up data
    ds_train = createDataset(
        ticker_list_csv_path=data_csv,
        root=data_root,
        lookback_window=lookback_window,
        prediction_horizon=prediction_horizon,
        split="train",
        target_column=target_column,
        feature_columns=feature_columns,
        download=False,
        force_regenerate=False,
    )
    ds_val = createDataset(
        ticker_list_csv_path=data_csv,
        root=data_root,
        lookback_window=lookback_window,
        prediction_horizon=prediction_horizon,
        split="val",
        target_column=target_column,
        feature_columns=feature_columns,
        download=False,
        force_regenerate=False,
    )
    ds_test = createDataset(
        ticker_list_csv_path=data_csv,
        root=data_root,
        lookback_window=lookback_window,
        prediction_horizon=prediction_horizon,
        split="test",
        target_column=target_column,
        feature_columns=feature_columns,
        download=False,
        force_regenerate=False,
    )

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # instantiate model with these params
    d_model = 128
    freq_k = 64
    merge_mode = "linear"
    merge_k = 2
    
    model = Hidformer(
        input_dim=len(feature_columns),
        pred_len=prediction_horizon,
        token_length=token_length,
        stride=token_length // 2,
        num_time_blocks=time_blocks,
        num_freq_blocks=freq_blocks,
        d_model=d_model,
        freq_k=freq_k,
        dropout=0.2,
        merge_mode=merge_mode,
        merge_k=merge_k,
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # early stopping stuff
    best_val_loss = float('inf')
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0
        for X, y in loader_train:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            # select target column
            if y.shape[-1] != pred.shape[-1]:
                idx = feature_columns.index(target_column)
                pred = pred[:, :, idx:idx+1]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(loader_train)

        # val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xv, yv in loader_val:
                Xv, yv = Xv.to(device), yv.to(device)
                pv = model(Xv)
                if yv.shape[-1] != pv.shape[-1]:
                    idx = feature_columns.index(target_column)
                    pv = pv[:, :, idx:idx+1]
                val_loss += criterion(pv, yv).item()
        val_loss /= len(loader_val)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # early stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
        else:
            stop_counter = stop_counter + 1 if 'stop_counter' in locals() else 1
        if stop_counter >= patience:
            break

    total_time = time.time() - start_time
    # test
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for Xt, yt in loader_test:
            Xt, yt = Xt.to(device), yt.to(device)
            pt = model(Xt)
            if yt.shape[-1] != pt.shape[-1]:
                idx = feature_columns.index(target_column)
                pt = pt[:, :, idx:idx+1]
            test_loss += criterion(pt, yt).item()
    test_loss /= len(loader_test)

    return {
        'time_blocks': time_blocks,
        'freq_blocks': freq_blocks,
        'token_length': token_length,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'train_time': total_time,
        'history': history
    }

def main():
    # stuff you can change
    DATA_CSV = './Hidformer-Hierarchal-dual-tower-tranfomer/src/tickers.csv'
    DATA_ROOT = './data/'
    LOOKBACK = 60
    PRED = 10
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
    TARGET = 'Close'

    BATCH = 32
    LR = 1e-4
    EPOCHS = 30
    PATIENCE = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device', device)

    # param lists
    time_blocks_list = [2, 4, 6]
    freq_blocks_list = [1, 2, 4]
    token_lengths = [8, 16, 24]

    results = []
    for tb in time_blocks_list:
        for fb in freq_blocks_list:
            for tl in token_lengths:
                print(f'running exp tb={tb}, fb={fb}, tl={tl}')
                res = run_experiment(tb, fb, tl,
                                     LOOKBACK, PRED,
                                     FEATURES, TARGET,
                                     BATCH, LR, EPOCHS, PATIENCE,
                                     DATA_CSV, DATA_ROOT,
                                     device)
                results.append({k: v for k, v in res.items() if k != 'history'})
                # optionally save history plots here

    df = pd.DataFrame(results)
    df.to_csv('sweep_results.csv', index=False)

    # now make 8 plots
    # 1. val loss vs time_blocks (avg over token len & freq)
    df_tb = df.groupby('time_blocks')['best_val_loss'].mean().reset_index()
    plt.figure()
    plt.plot(df_tb['time_blocks'], df_tb['best_val_loss'], marker='o')
    plt.title('Val Loss vs Time Blocks')
    plt.xlabel('Time Blocks')
    plt.ylabel('Val Loss')
    plt.savefig('plot_val_vs_timeblocks.png')

    # 2. val loss vs freq_blocks
    df_fb = df.groupby('freq_blocks')['best_val_loss'].mean().reset_index()
    plt.figure()
    plt.plot(df_fb['freq_blocks'], df_fb['best_val_loss'], marker='o')
    plt.title('Val Loss vs Freq Blocks')
    plt.xlabel('Freq Blocks')
    plt.ylabel('Val Loss')
    plt.savefig('plot_val_vs_freqblocks.png')

    # 3. val loss vs token_length
    df_tl = df.groupby('token_length')['best_val_loss'].mean().reset_index()
    plt.figure()
    plt.plot(df_tl['token_length'], df_tl['best_val_loss'], marker='o')
    plt.title('Val Loss vs Token Length')
    plt.xlabel('Token Length')
    plt.ylabel('Val Loss')
    plt.savefig('plot_val_vs_tokenlength.png')

    # 4. heatmap time_blocks vs freq_blocks
    pivot_tb_fb = df.pivot_table(values='best_val_loss', index='time_blocks', columns='freq_blocks')
    plt.figure()
    plt.imshow(pivot_tb_fb.values, aspect='auto')
    plt.title('Heatmap Val Loss (time_blocks vs freq_blocks)')
    plt.xlabel('Freq Blocks')
    plt.ylabel('Time Blocks')
    plt.colorbar()
    plt.savefig('heatmap_tb_fb.png')

    # 5. heatmap time_blocks vs token_length
    pivot_tb_tl = df.pivot_table(values='best_val_loss', index='time_blocks', columns='token_length')
    plt.figure()
    plt.imshow(pivot_tb_tl.values, aspect='auto')
    plt.title('Heatmap Val Loss (time_blocks vs token_length)')
    plt.xlabel('Token Length')
    plt.ylabel('Time Blocks')
    plt.colorbar()
    plt.savefig('heatmap_tb_tl.png')

    # 6. heatmap freq_blocks vs token_length
    pivot_fb_tl = df.pivot_table(values='best_val_loss', index='freq_blocks', columns='token_length')
    plt.figure()
    plt.imshow(pivot_fb_tl.values, aspect='auto')
    plt.title('Heatmap Val Loss (freq_blocks vs token_length)')
    plt.xlabel('Token Length')
    plt.ylabel('Freq Blocks')
    plt.colorbar()
    plt.savefig('heatmap_fb_tl.png')

    # 7. training time by config
    plt.figure()
    plt.bar(range(len(df)), df['train_time'])
    plt.title('Training Time per Config')
    plt.xlabel('Experiment Index')
    plt.ylabel('Time (s)')
    plt.savefig('bar_train_time.png')

    # 8. test loss vs best val loss
    plt.figure()
    plt.scatter(df['best_val_loss'], df['test_loss'])
    plt.title('Test Loss vs Best Val Loss')
    plt.xlabel('Best Val Loss')
    plt.ylabel('Test Loss')
    plt.savefig('scatter_test_vs_val.png')

    print('done, results saved and plots generated')

if __name__ == '__main__':
    main()
