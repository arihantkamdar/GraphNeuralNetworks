import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split


def prepare_dataset():
    df = pd.read_csv("/home/arihant/MSC/Data/formatted_data_clean_pct_change_no_vol.csv")
    cols = df.columns
    for col in cols:
        df[col] = StandardScaler().fit_transform(df[[col]])
    print(df)
    assets = ['BSE SENSEX', 'Crude', 'Dow Jones Industrial Average',
              'Euronext 100', 'FTSE 100 Index', 'Gold', 'NASDAQ Composite',
              'NYSE Composite', 'Nifty 50', 'Nikkei 225', 'S&P 500 Index', 'SSE Composite Index']

    # ohlc = ['Open', 'High', 'Low', 'Close']
    ohlc = ['Close']
    asset_feature_data = []
    for asset in assets:
        print(asset)
        cols = [f"{asset} {f}" for f in ohlc]
        raw = df[cols].values  # shape: [T, 4]
        asset_feature_data.append(raw)
    X_all = np.stack(asset_feature_data, axis=1)  # [T, num_assets, num_features]


    print(X_all.shape)
    Y_all = X_all[1:]
    X_all = X_all[:-1]
    print(X_all.shape)
    print(Y_all.shape)
    return  X_all[-1000:], Y_all[-1000:]
    # return X_all, Y_all



import torch
from torch.utils.data import Dataset

class MTGNNDataset(Dataset):
    def __init__(self, X, Y, seq_len=12, pred_len=1):
        """
        X: np.array of shape [T, N, F] — features
        Y: np.array of shape [T, N, F] — targets
        seq_len: input sequence length
        pred_len: prediction horizon (how far in future to predict)
        """
        assert X.shape == Y.shape
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = X.shape[0] - seq_len - pred_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input: [seq_len, N, F]
        x = self.X[idx: idx + self.seq_len]

        # Target: [N] — next-step 'Close' values per node
        y = self.Y[idx + self.seq_len]  # assuming F=1, use [:, 3] if Close is 4th feature
        x = torch.FloatTensor(x).permute(1, 2, 0)  # [N, F, T]
        return x, torch.FloatTensor(y)
        # return torch.FloatTensor(x), torch.FloatTensor(y)



if __name__ == "__main__":

    seq_len = 12
    pred_len = 1
    X_all, Y_all = prepare_dataset()
    dataset = MTGNNDataset(X_all, Y_all, seq_len=seq_len, pred_len=pred_len)

    # Split: 70% train, 15% val, 15% test
    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)