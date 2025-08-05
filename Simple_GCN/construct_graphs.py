from itertools import combinations

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.utils import add_self_loops

# -----------------------
# Load and preprocess data
# -----------------------
from torch_geometric.utils import degree

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import numpy as np


def construct_simple_correlation_graphs(
        df,
        corr_window=10,
        corr_threshold=0.6,
        self_loop=True
):
    graphs = []

    # List of assets
    assets = ['BSE SENSEX', 'Crude', 'Dow Jones Industrial Average',
              'Euronext 100', 'FTSE 100 Index', 'Gold', 'NASDAQ Composite',
              'NYSE Composite', 'Nifty 50', 'Nikkei 225', 'S&P 500 Index', 'SSE Composite Index']

    # OHLC features to extract
    ohlc = ['Open', 'High', 'Low', 'Close']
    num_assets = len(assets)
    num_features = len(ohlc)

    # Prepare OHLC data per asset and scale
    asset_feature_data = []
    for asset in assets:
        cols = [f"{asset} {f}" for f in ohlc]
        raw = df[cols].values  # shape: [T, 4]
        scaled = StandardScaler().fit_transform(raw)
        asset_feature_data.append(scaled)

    # Shape: [T, num_assets, num_features]
    X_all = np.stack(asset_feature_data, axis=1)

    # Prepare Close price data for correlation
    close_cols = [f"{a} Close" for a in assets]
    df_close = df[close_cols]

    for t in range(corr_window, len(df_close) - 1):
        # Step 1: Correlation matrix of Close prices
        window = df_close.iloc[t - corr_window:t]
        corr_matrix = window.corr().fillna(0).values

        edge_index, edge_weights = [], []
        for i, j in combinations(range(num_assets), 2):
            corr = corr_matrix[i, j]
            if abs(corr) >= corr_threshold:
                corr = (corr+1)/2 # range goes from -1 to 1 to 0 to 1
                edge_index += [[i, j], [j, i]]
                edge_weights += [abs(corr), abs(corr)]
            edge_index += [[i, j], [j, i]]
            edge_weights += [abs(corr), abs(corr)]



        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        edge_attr[torch.isnan(edge_attr)] = 0  # Replace NaNs with 0
        edge_attr[torch.isinf(edge_attr)] = 0  # Replace Infs with 0

        # edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0, neginf=0)

        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_assets)

        # Node features at time t
        x = torch.tensor(X_all[t], dtype=torch.float)  # shape: [num_assets, 4]

        # Target: Next day's Close prices
        y = torch.tensor(df_close.iloc[t + 1].values, dtype=torch.float).unsqueeze(1)  # [num_assets, 1]

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    return graphs



def construct_simple_correlation_graphs_no_ss(
        df,
        corr_window=10,
        corr_threshold=0.6,
        self_loop=True
):
    graphs = []

    # List of assets
    assets = ['BSE SENSEX', 'Crude', 'Dow Jones Industrial Average',
              'Euronext 100', 'FTSE 100 Index', 'Gold', 'NASDAQ Composite',
              'NYSE Composite', 'Nifty 50', 'Nikkei 225', 'S&P 500 Index', 'SSE Composite Index']

    # OHLC features to extract
    ohlc = ['Open', 'High', 'Low', 'Close']
    num_assets = len(assets)
    num_features = len(ohlc)

    # Prepare OHLC data per asset (without scaling)
    asset_feature_data = []
    for asset in assets:
        cols = [f"{asset} {f}" for f in ohlc]
        raw = df[cols].values  # shape: [T, 4]
        asset_feature_data.append(raw)

    # Shape: [T, num_assets, num_features]
    X_all = np.stack(asset_feature_data, axis=1)

    # Prepare Close price data for correlation
    close_cols = [f"{a} Close" for a in assets]
    df_close = df[close_cols]

    for t in range(corr_window, len(df_close) - 1):
        # Step 1: Correlation matrix of Close prices
        window = df_close.iloc[t - corr_window:t]
        corr_matrix = window.corr().fillna(0).values

        edge_index, edge_weights = [], []
        for i, j in combinations(range(num_assets), 2):
            corr = corr_matrix[i, j]
            if abs(corr) >= corr_threshold:
                corr = (corr + 1) / 2  # Normalize from [-1,1] to [0,1]
                edge_index += [[i, j], [j, i]]
                edge_weights += [abs(corr), abs(corr)]

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        edge_attr[torch.isnan(edge_attr)] = 0
        edge_attr[torch.isinf(edge_attr)] = 0

        if self_loop:
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_assets)

        # Node features at time t
        x = torch.tensor(X_all[t], dtype=torch.float)  # shape: [num_assets, 4]

        # Target: Next day's Close prices
        y = torch.tensor(df_close.iloc[t + 1].values, dtype=torch.float).unsqueeze(1)  # [num_assets, 1]

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    return graphs

def graphs_wrapper(window, path):
    df = pd.read_csv('/Data/formatted_data_clean_pct_change.csv')
    cols = df.columns
    for col in cols:
        df[col] = StandardScaler().fit_transform(df[col])
    graphs = construct_simple_correlation_graphs_no_ss(df)
    new_graphs = []
    for idx in range(graphs[window:], len(graphs)):
        temp_x = []
        for g in graphs[idx:idx+window]:
            temp_x



if __name__ == "__main__":
    df = pd.read_csv('/Data/formatted_data_clean_pct_change.csv')
    cols = df.columns
    for col in cols:
        df[col] = StandardScaler().fit_transform(df[col])
    graphs = construct_simple_correlation_graphs_no_ss(df)

    print(graphs[0].x, graphs[0].y, graphs[0].edge_attr, graphs[0].edge_index)
    print(graphs[0].x, graphs[0].y, graphs[0].edge_attr, graphs[0].edge_index)