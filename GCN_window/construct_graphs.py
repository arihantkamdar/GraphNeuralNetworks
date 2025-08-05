import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def construct_correlation_graphs_with_temporal_x(
        df, corr_window=10, seq_len=5, corr_threshold=0.6, self_loop=True
):
    graphs = []
    assets = [
        'BSE SENSEX', 'Crude', 'Dow Jones Industrial Average',
        'Euronext 100', 'FTSE 100 Index', 'Gold', 'NASDAQ Composite',
        'NYSE Composite', 'Nifty 50', 'Nikkei 225', 'S&P 500 Index', 'SSE Composite Index'
    ]
    ohlc = ['Open', 'High', 'Low', 'Close']
    num_assets = len(assets)
    num_features = len(ohlc)

    # Validate dataset
    required_cols = [f"{asset} {f}" for asset in assets for f in ohlc]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    if not df.index.is_monotonic_increasing:
        logger.warning("Dataset not sorted by date; sorting now")
        df = df.sort_index()

    # Scale OHLC features across assets
    ohlc_cols = required_cols
    scaler = StandardScaler()
    df[ohlc_cols] = scaler.fit_transform(df[ohlc_cols])

    # Full OHLC tensor: [T, num_assets, num_features]
    asset_feature_data = []
    for asset in assets:
        cols = [f"{asset} {f}" for f in ohlc]
        raw = df[cols].values  # shape: [T, 4]
        asset_feature_data.append(raw)
    X_all = np.stack(asset_feature_data, axis=1)  # [T, num_assets, num_features]

    # Close prices for correlation
    close_cols = [f"{a} Close" for a in assets]
    df_close = df[close_cols]

    start_t = max(corr_window, seq_len)
    for t in range(start_t, len(df_close) - 1):
        # Correlation matrix for edge construction
        corr_window_df = df_close.iloc[t - corr_window:t]
        corr_matrix = corr_window_df.corr().fillna(0).values

        edge_index, edge_weights = [], []
        for i, j in combinations(range(num_assets), 2):
            corr = corr_matrix[i, j]
            if abs(corr) >= corr_threshold:
                norm_corr = (corr + 1) / 2
                edge_index += [[i, j], [j, i]]
                edge_weights += [abs(norm_corr), abs(norm_corr)]

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        edge_attr[torch.isnan(edge_attr)] = 0
        edge_attr[torch.isinf(edge_attr)] = 0

        if self_loop:
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_assets)

        # Node features: [num_assets, seq_len, num_features]
        x_window = X_all[t - seq_len:t]  # [seq_len, num_assets, num_features]
        x = torch.tensor(x_window.transpose(1, 0, 2), dtype=torch.float)  # [num_assets, seq_len, num_features]

        # Target: Next-day Close prices
        y = torch.tensor(df_close.iloc[t + 1].values, dtype=torch.float).unsqueeze(1)  # [num_assets, 1]

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        logger.debug(f"Graph at t={t}: {edge_index.shape[1]//2} edges")

    logger.info(f"Generated {len(graphs)} graphs, Nodes: {num_assets}, Features: {x.shape[1:]}")
    logger.info(f"Sample target mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")
    logger.info(f"Sample edge count: {edge_index.shape[1] // 2}")
    return graphs

def construct_graphs(corr_window, seq_len, corr_threshold):
    df = pd.read_csv('/home/arihant/MSC/Data/formatted_data_clean_pct_change_no_vol.csv')
    graphs = construct_correlation_graphs_with_temporal_x(
        df=df, corr_window=corr_window, seq_len=seq_len, corr_threshold=corr_threshold, self_loop=True
    )
    return graphs