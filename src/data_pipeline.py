import os
import torch
from embedding import download_data
from preprocessing import preprocess_prices, create_sequences, TimeSeriesDataset
from torch.utils.data import DataLoader, random_split

class DataPipeline:
    def __init__(
        self,
        tickers,
        start_date,
        end_date,
        seq_len,
        pred_len,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=32,
        interval='1d',
        data_dir='data'
    ):
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

    def prepare_data(self):
        fp = os.path.join(self.data_dir, 'prices.csv')
        if os.path.exists(fp):
            import pandas as pd
            price_df = pd.read_csv(fp, index_col=0, parse_dates=True)
        else:
            price_df = download_data(
                self.tickers,
                self.start_date,
                self.end_date,
                interval=self.interval
            )
            price_df.to_csv(fp)

        data_array, self.scaler = preprocess_prices(price_df)
        X, y = create_sequences(data_array, self.seq_len, self.pred_len)
        dataset = TimeSeriesDataset(X, y)

        total = len(dataset)
        test_n = int(total * self.test_ratio)
        val_n = int(total * self.val_ratio)
        train_n = total - val_n - test_n

        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset,
            [train_n, val_n, test_n],
            generator=torch.Generator().manual_seed(42)
        )

    def get_loaders(self):
        assert hasattr(self, 'train_ds'), 'Call prepare_data() first'
        return (
            DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True),
            DataLoader(self.val_ds,   batch_size=self.batch_size, shuffle=False),
            DataLoader(self.test_ds,  batch_size=self.batch_size, shuffle=False)
        )
