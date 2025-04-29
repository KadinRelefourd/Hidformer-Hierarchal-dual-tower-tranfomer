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

def run_experiment(time_blocks=6, freq_blocks=2, token_length=16,
                   lookback_window=60, prediction_horizon=20,
                   feature_columns=None, target_column='Close',
                   batch_size=32, learning_rate=1e-4, epochs=50,
                   patience=10, data_csv=None, data_root='./data',
                   device=None, model_save_path='models',
                   d_model=128, freq_k=64):
    """
    Train and evaluate a HiDFORM model for stock market prediction.
    
    Args:
        time_blocks: Number of time blocks in the time tower
        freq_blocks: Number of frequency blocks in the frequency tower
        token_length: Length of each token segment
        lookback_window: Number of time steps to look back
        prediction_horizon: Number of time steps to predict
        feature_columns: List of feature columns to use
        target_column: Target column to predict
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        epochs: Maximum number of epochs
        patience: Early stopping patience
        data_csv: Path to CSV with ticker symbols
        data_root: Root directory for data
        device: Device to run on (cuda/cpu)
        model_save_path: Directory to save models
        d_model: Main dimension of model's hidden states
        freq_k: Low-rank dimension for LinearAttention
    
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*80}\nRunning experiment with {time_blocks} time blocks, {freq_blocks} frequency blocks\n{'='*80}")
    
    # Default feature columns if None
    if feature_columns is None:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Make sure target is included in features
    if target_column not in feature_columns:
        feature_columns.append(target_column)
    
    # Create model save directory if it doesn't exist
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Create unique model name based on parameters
    model_name = f"hidformer_t{time_blocks}_f{freq_blocks}_tok{token_length}_lb{lookback_window}_ph{prediction_horizon}"
    
    # Prepare data
    print("Loading datasets...")
    ds_train = createDataset(
        ticker_list_csv_path=data_csv, root=data_root,
        lookback_window=lookback_window, prediction_horizon=prediction_horizon,
        split="train", target_column=target_column,
        feature_columns=feature_columns, download=True,
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

    print(f"Dataset sizes - Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")
    
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # Set up model
    print(f"Initializing HiDFORM model with {time_blocks} time blocks and {freq_blocks} frequency blocks...")
    model = Hidformer(
        input_dim=len(feature_columns), 
        pred_len=prediction_horizon,
        token_length=token_length, 
        stride=token_length // 2,  # Use 50% overlap for better sequence modeling
        num_time_blocks=time_blocks,  # As requested: 6 time blocks
        num_freq_blocks=freq_blocks,
        d_model=d_model, 
        freq_k=freq_k,
        dropout=0.2, 
        merge_mode="linear", 
        merge_k=2
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop with early stopping
    print("Starting training...")
    best_model_path = os.path.join(model_save_path, f"{model_name}_best.pth")
    best_val_loss = float('inf')
    stop_counter = 0
    train_losses, val_losses = [], []
    
    start_time = time.time()
    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        for X, y in loader_train:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            
            # Handle case where model predicts all features but we only want target
            if isinstance(target_column, str) and y.shape[-1] != pred.shape[-1]:
                idx = feature_columns.index(target_column)
                pred = pred[:, :, idx:idx+1]
            
            loss = criterion(pred, y)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(loader_train)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in loader_val:
                X_val, y_val = X_val.to(device), y_val.to(device)
                pred_val = model(X_val)
                
                # Handle case where model predicts all features but we only want target
                if isinstance(target_column, str) and y_val.shape[-1] != pred_val.shape[-1]:
                    idx = feature_columns.index(target_column)
                    pred_val = pred_val[:, :, idx:idx+1]
                
                val_loss += criterion(pred_val, y_val).item()
        
        val_loss /= len(loader_val)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"√ New best model saved (val_loss: {val_loss:.6f})")
        else:
            stop_counter += 1
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{epochs} - Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, "
              f"Time: {epoch_time:.2f}s, EarlyStopping: {stop_counter}/{patience}")
        
        if stop_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Load best model and evaluate on test set
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Collect predictions and ground truth for visualization
    all_preds = []
    all_targets = []
    test_loss = 0
    
    with torch.no_grad():
        for X_test, y_test in loader_test:
            X_test, y_test = X_test.to(device), y_test.to(device)
            pred_test = model(X_test)
            
            # Handle case where model predicts all features but we only want target
            if isinstance(target_column, str) and y_test.shape[-1] != pred_test.shape[-1]:
                idx = feature_columns.index(target_column)
                pred_test = pred_test[:, :, idx:idx+1]
            
            test_loss += criterion(pred_test, y_test).item()
            
            all_preds.append(pred_test.cpu().numpy())
            all_targets.append(y_test.cpu().numpy())
    
    test_loss /= len(loader_test)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"Test loss: {test_loss:.6f}")
    
    # Save results
    result = {
        'time_blocks': time_blocks,
        'freq_blocks': freq_blocks,
        'token_length': token_length,
        'train_loss': train_losses[-1],
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_path': best_model_path,
        'all_preds': all_preds,
        'all_targets': all_targets,
        'lookback_window': lookback_window,
        'prediction_horizon': prediction_horizon
    }
    
    return result


def plot_learning_curves(train_losses, val_losses):
    """Plot training and validation learning curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_actual_vs_pred(true_series, pred_series, config_name='Hidformer'):
    """
    Plot actual vs predicted values.
    
    Args:
        true_series: Ground truth series (numpy array)
        pred_series: Predicted series (numpy array)
        config_name: Configuration name for the title
    """
    # Get random samples to plot
    samples = min(3, true_series.shape[0])
    indices = np.random.choice(true_series.shape[0], samples, replace=False)
    
    plt.figure(figsize=(15, 4 * samples))
    
    for i, idx in enumerate(indices):
        plt.subplot(samples, 1, i+1)
        ts = true_series[idx, :, 0]
        ps = pred_series[idx, :, 0]
        
        # Calculate trend direction
        trend = "Mixed"
        if ts[-1] > ts[0] * 1.02:  # 2% increase threshold
            trend = "Upward"
        elif ts[-1] < ts[0] * 0.98:  # 2% decrease threshold
            trend = "Downward"
        
        # Calculate error metrics
        mse = np.mean((ts - ps) ** 2)
        mae = np.mean(np.abs(ts - ps))
        
        plt.plot(ts, 'b-', label='Actual', linewidth=2)
        plt.plot(ps, 'r--', label='Predicted', linewidth=2)
        plt.title(f'{trend} Trend Prediction (Sample {idx}) - MSE: {mse:.4f}, MAE: {mae:.4f}')
        plt.xlabel('Time step')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'prediction_comparison_{config_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_trend_performance(all_targets, all_preds, threshold=0.02):
    """
    Analyze model performance by trend direction.
    
    Args:
        all_targets: Ground truth values
        all_preds: Predicted values
        threshold: Threshold for determining trend direction (percentage change)
        
    Returns:
        DataFrame with performance metrics by trend
    """
    results = []
    
    for i in range(all_targets.shape[0]):
        ts = all_targets[i, :, 0]
        ps = all_preds[i, :, 0]
        
        # Determine trend
        if ts[-1] > ts[0] * (1 + threshold):
            trend = "Upward"
        elif ts[-1] < ts[0] * (1 - threshold):
            trend = "Downward"
        else:
            trend = "Mixed"
            
        # Calculate metrics
        mse = np.mean((ts - ps) ** 2)
        mae = np.mean(np.abs(ts - ps))
        
        # Direction accuracy (did the model predict the right direction?)
        actual_direction = 1 if ts[-1] > ts[0] else (-1 if ts[-1] < ts[0] else 0)
        pred_direction = 1 if ps[-1] > ps[0] else (-1 if ps[-1] < ps[0] else 0)
        direction_match = actual_direction == pred_direction
        
        results.append({
            'trend': trend,
            'mse': mse,
            'mae': mae,
            'direction_match': direction_match
        })
    
    df = pd.DataFrame(results)
    
    # Aggregate by trend
    trend_summary = df.groupby('trend').agg({
        'mse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'direction_match': 'mean'  # This gives direction accuracy percentage
    }).reset_index()
    
    # Also get overall stats
    overall = {
        'trend': 'Overall',
        'mse_mean': df['mse'].mean(),
        'mse_std': df['mse'].std(),
        'mae_mean': df['mae'].mean(),
        'mae_std': df['mae'].std(),
        'direction_match_mean': df['direction_match'].mean()
    }
    
    # Convert to nice dataframe
    summary_df = pd.DataFrame({
        'Trend': list(trend_summary['trend']) + [overall['trend']],
        'MSE': [f"{row[('mse', 'mean')]:.4f} ± {row[('mse', 'std')]:.4f}" for _, row in trend_summary.iterrows()] + 
               [f"{overall['mse_mean']:.4f} ± {overall['mse_std']:.4f}"],
        'MAE': [f"{row[('mae', 'mean')]:.4f} ± {row[('mae', 'std')]:.4f}" for _, row in trend_summary.iterrows()] + 
               [f"{overall['mae_mean']:.4f} ± {overall['mae_std']:.4f}"],
        'Direction Accuracy': [f"{row[('direction_match', 'mean')]*100:.1f}%" for _, row in trend_summary.iterrows()] + 
                             [f"{overall['direction_match_mean']*100:.1f}%"]
    })
    
    # Count samples per trend
    counts = df['trend'].value_counts().to_dict()
    print(f"Sample counts by trend: {counts}")
    
    return summary_df


def plot_trend_examples(all_targets, all_preds, n_samples=1):
    """
    Plot examples of upward, downward, and mixed trends.
    
    Args:
        all_targets: Ground truth values
        all_preds: Predicted values
        n_samples: Number of samples to plot per trend
    """
    # Classify all samples by trend
    trends = {'Upward': [], 'Downward': [], 'Mixed': []}
    
    threshold = 0.02  # 2% change threshold for trend classification
    
    for i in range(all_targets.shape[0]):
        ts = all_targets[i, :, 0]
        
        if ts[-1] > ts[0] * (1 + threshold):
            trends['Upward'].append(i)
        elif ts[-1] < ts[0] * (1 - threshold):
            trends['Downward'].append(i)
        else:
            trends['Mixed'].append(i)
    
    plt.figure(figsize=(15, 12))
    
    # For each trend, plot n_samples examples
    for i, (trend, indices) in enumerate(trends.items()):
        if indices:  # If we have examples of this trend
            # Select samples (all if fewer than n_samples)
            sample_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
            
            for j, idx in enumerate(sample_indices):
                plt.subplot(3, n_samples, i*n_samples + j + 1)
                
                ts = all_targets[idx, :, 0]
                ps = all_preds[idx, :, 0]
                
                plt.plot(ts, 'b-', label='Actual', linewidth=2)
                plt.plot(ps, 'r--', label='Predicted', linewidth=2)
                plt.title(f'{trend} Trend Example')
                plt.xlabel('Time step')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trend_examples.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Set up configurations
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Data configuration
    DATA_CSV = os.path.join(BASE_DIR, 'tickers.csv')
    DATA_ROOT = os.path.join(BASE_DIR, 'data')
    
    # Model hyperparameters
    LOOKBACK = 60           # Input window size
    PRED_HORIZON = 20       # Prediction horizon
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']  # Features to use
    TARGET = 'Close'        # Target to predict
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    PATIENCE = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Model configuration
    TIME_BLOCKS = 6  # As requested
    FREQ_BLOCKS = 2  # Keep frequency blocks at 2
    TOKEN_LENGTH = 16  # Standard token length
    
    # Run the main experiment
    result = run_experiment(
        time_blocks=TIME_BLOCKS,
        freq_blocks=FREQ_BLOCKS,
        token_length=TOKEN_LENGTH,
        lookback_window=LOOKBACK,
        prediction_horizon=PRED_HORIZON,
        feature_columns=FEATURES,
        target_column=TARGET,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        patience=PATIENCE,
        data_csv=DATA_CSV,
        data_root=DATA_ROOT,
        device=device,
        model_save_path='models'
    )
    
    # Plot learning curves
    plot_learning_curves(result['train_losses'], result['val_losses'])
    
    # Plot prediction examples
    plot_actual_vs_pred(
        result['all_targets'], 
        result['all_preds'], 
        f"HiDFORM_T{TIME_BLOCKS}_F{FREQ_BLOCKS}"
    )
    
    # Analyze performance by trend
    trend_summary = analyze_trend_performance(result['all_targets'], result['all_preds'])
    print("\nPerformance by trend:")
    print(trend_summary)
    
    # Save trend summary
    trend_summary.to_csv('trend_performance.csv', index=False)
    
    # Plot trend examples
    plot_trend_examples(result['all_targets'], result['all_preds'], n_samples=2)
    
    print("\nExperiment completed successfully!")
    
    # Optional: Compare different configurations
    print("\nWould you like to run a hyperparameter comparison? (y/n)")
    response = input().lower()
    
    if response == 'y':
        # Compare different token lengths
        token_lengths = [8, 16, 32]
        token_results = []
        
        for tl in token_lengths:
            print(f"\nRunning experiment with token_length={tl}...")
            res = run_experiment(
                time_blocks=TIME_BLOCKS,
                freq_blocks=FREQ_BLOCKS,
                token_length=tl,
                lookback_window=LOOKBACK,
                prediction_horizon=PRED_HORIZON,
                feature_columns=FEATURES,
                target_column=TARGET,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                epochs=EPOCHS,
                patience=PATIENCE,
                data_csv=DATA_CSV,
                data_root=DATA_ROOT,
                device=device,
                model_save_path='models'
            )
            token_results.append({
                'token_length': tl,
                'test_loss': res['test_loss'],
                'best_val_loss': res['best_val_loss']
            })
            
            # Plot for this configuration
            plot_actual_vs_pred(
                res['all_targets'], 
                res['all_preds'], 
                f"TokenLen_{tl}"
            )
            
        # Save token length comparison
        token_df = pd.DataFrame(token_results)
        token_df.to_csv('token_length_comparison.csv', index=False)
        print("\nToken length comparison:")
        print(token_df)


if __name__ == '__main__':
    main()

