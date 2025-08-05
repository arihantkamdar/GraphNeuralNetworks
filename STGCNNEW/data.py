from itertools import combinations
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.utils import add_self_loops

def construct_correlation_graphs_with_temporal_x(
        df,
        corr_window=10,
        window_len=20,
        corr_threshold=0.5,
        self_loop=True
):
    graphs = []
    assets = ['BSE SENSEX', 'Crude', 'Dow Jones Industrial Average',
              'Euronext 100', 'FTSE 100 Index', 'Gold', 'NASDAQ Composite',
              'NYSE Composite', 'Nifty 50', 'Nikkei 225', 'S&P 500 Index', 'SSE Composite Index']
    ohlc = ['Open', 'High', 'Low', 'Close']
    num_assets = len(assets)
    num_features = len(ohlc)

    # Full OHLC tensor: [T, num_assets, num_features]
    asset_feature_data = []
    for asset in assets:
        cols = [f"{asset} {f}" for f in ohlc]
        raw = df[cols].values
        asset_feature_data.append(raw)
    X_all = np.stack(asset_feature_data, axis=1)

    # Close prices for correlation
    close_cols = [f"{a} Close" for a in assets]
    df_close = df[close_cols]

    start_t = max(corr_window, window_len)
    for t in range(start_t, len(df_close) - 1):
        corr_window_df = df_close.iloc[t - corr_window:t]
        corr_matrix = corr_window_df.corr().fillna(0).values

        edge_index, edge_weights = [], []
        for i, j in combinations(range(num_assets), 2):
            corr = corr_matrix[i, j]
            if abs(corr) >= corr_threshold:
                edge_index += [[i, j], [j, i]]
                corr = (corr+1)/2
                edge_weights += [corr, corr]  # Use raw correlation

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        edge_attr[torch.isnan(edge_attr)] = 0
        edge_attr[torch.isinf(edge_attr)] = 0


        if self_loop:
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_assets)

        x_window = X_all[t - window_len:t]
        x = torch.tensor(x_window.transpose(1, 0, 2), dtype=torch.float)

        y = torch.tensor(df_close.iloc[t + 1].values, dtype=torch.float).unsqueeze(1)
        # print(x)
        # print("*******")
        # print(edge_attr)
        # print("*******")
        # print(edge_index)
        # print("*******")
        # print(y)
        # # exit()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.x = data.x.permute(0, 2, 1)
        # print(x)

        # exit()
        graphs.append(data)

    return graphs

def construct_graphs(window, seq_l, corr_threshold):
    df = pd.read_csv('/home/arihant/MSC/Data/formatted_data_clean_pct_change_no_vol.csv')
    cols = df.columns
    for col in cols:
        df[col] = StandardScaler().fit_transform(df[[col]])

    graphs = construct_correlation_graphs_with_temporal_x(
        df=df, corr_threshold=corr_threshold, self_loop=True, corr_window=window, window_len=seq_l
    )
    # Check for NaN or Inf in graphs
    for i, graph in enumerate(graphs):
        if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
            print(f"Warning: NaN or Inf in x for graph {i}")
        if torch.isnan(graph.y).any() or torch.isinf(graph.y).any():
            print(f"Warning: NaN or Inf in y for graph {i}")
        if torch.isnan(graph.edge_attr).any() or torch.isinf(graph.edge_attr).any():
            print(f"Warning: NaN or Inf in edge_attr for graph {i}")
    return graphs