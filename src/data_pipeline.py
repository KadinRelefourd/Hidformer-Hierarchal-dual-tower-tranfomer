import os
from embedding import download_data
from preprocessing import preprocess_prices, create_sequences, TimeSeriesDataset
from torch.utils.data import DataLoader, random_split


class DataPipeline:
    """
    Orchestrates downloading, preprocessing, splitting, and DataLoader creation.
    """
    def __init__(self,
                 tickers,
                 start_date,
                 end_date,
                 seq_len,
                 pred_len,
                 val_ratio=0.1,
                 test_ratio=0.1,
                 batch_size=32,
                 interval='1d',
                 data_dir='data'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.interval = interval
        self.data_dir = data_dir

        os.makedirs(self.data_dir, exist_ok=True)

        self.train_ds = self.val_ds = self.test_ds = None

    def prepare_data(self):
        # 1) Download raw prices
        csv_path = os.path.join(self.data_dir, 'prices.csv')
        if os.path.exists(csv_path):
            import pandas as pd
            price_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            price_df = download_data(self.tickers,
                                     self.start_date,
                                     self.end_date,
                                     interval=self.interval)
            price_df.to_csv(csv_path)

        # 2) Preprocess: fill, scale
        data_array, self.scaler = preprocess_prices(price_df)

        # 3) Create sliding-window sequences
        X, y = create_sequences(data_array, self.seq_len, self.pred_len)

        # 4) Build Dataset
        dataset = TimeSeriesDataset(X, y)

        # 5) Split into train/val/test
        total_len = len(dataset)
        test_len = int(total_len * self.test_ratio)
        val_len = int(total_len * self.val_ratio)
        train_len = total_len - val_len - test_len
        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
        )

    def get_loaders(self):
        assert self.train_ds is not None, "Call prepare_data() first"
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Example usage
    tickers = ['MSFT', 'AAPL', 'NVDA', 'TSM']
    dp = DataPipeline(tickers,
                      start_date='2020-01-01',
                      end_date='2023-12-31',
                      seq_len=60,
                      pred_len=10,
                      val_ratio=0.1,
                      test_ratio=0.1,
                      batch_size=64)
    dp.prepare_data()
    train_loader, val_loader, test_loader = dp.get_loaders()
    print(f"#train: {len(dp.train_ds)}, #val: {len(dp.val_ds)}, #test: {len(dp.test_ds)}")
