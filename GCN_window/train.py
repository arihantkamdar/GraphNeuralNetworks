import logging
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from gcn import GCN_LSTM_WeightedEdges
from construct_graphs import construct_graphs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
    filename='huber2_train.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0

def train(train_loader, model, optimizer, loss_fn):
    model.train()
    total_loss = 0
    y_true_all, y_pred_all = [], []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, batch=data.batch)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        y_true_all.append(data.y.detach())
        y_pred_all.append(out.detach())
    y_true_all = torch.cat(y_true_all).cpu().numpy().flatten()
    y_pred_all = torch.cat(y_pred_all).cpu().numpy().flatten()
    r2 = r2_score(y_true_all, y_pred_all)
    final_loss = total_loss / len(train_loader.dataset)
    return final_loss, r2, y_true_all, y_pred_all

def evaluate(test_loader, model, loss_fn):
    model.eval()
    total_loss = 0
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, batch=data.batch)
            loss = loss_fn(out, data.y)
            total_loss += loss.item()
            y_true_all.append(data.y.detach())
            y_pred_all.append(out.detach())
        y_true_all = torch.cat(y_true_all).cpu().numpy().flatten()
        y_pred_all = torch.cat(y_pred_all).cpu().numpy().flatten()
        r2 = r2_score(y_true_all, y_pred_all)
        final_loss = total_loss / len(test_loader.dataset)
    return final_loss, r2, y_true_all, y_pred_all

def plots(epoch, train_losses, val_losses, val_r2s, train_r2s, y_true, y_pred, name):
    plt.figure(figsize=(15, 5))
    epochs = range(len(train_losses))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Loss')
    # plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    # plt.plot(epochs, val_r2s, label='Validation R2')
    plt.plot(epochs, train_r2s, label='R2')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.title('R2 Score over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(50), y_true[:50], label='true')
    plt.plot(range(50), y_pred[:50], label='pred')

    # plt.scatter(y_true, y_pred, alpha=0.5)
    # plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')

    plt.tight_layout()
    plt.savefig(f'images/{name}.png')
    plt.close()

def runner(corr_window, seq_len, corr_threshold):
    graphs = construct_graphs(corr_window=corr_window, seq_len=seq_len, corr_threshold=corr_threshold)
    graphs = graphs[:1000]
    random.seed(42)
    split = int(0.8 * len(graphs))  # Temporal split
    train_graphs = graphs[:split]
    test_graphs = graphs[split:]

    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    model = GCN_LSTM_WeightedEdges(
        in_features=4, seq_len=seq_len, gcn_hidden_dim=16, lstm_hidden_dim=16, out_features=1, lstm_layers=1
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # early_stopping = EarlyStopping(patience=10, verbose=True)

    logging.info(f"Device: {device}")
    logging.info(f"Corr_window: {corr_window}, Seq_len: {seq_len}, Corr_threshold: {corr_threshold}, LR: 0.001")
    logging.info(f"Number of training graphs: {len(train_graphs)}, test graphs: {len(test_graphs)}")

    loss_train, loss_val = [], []
    r2_train, r2_val = [], []
    name = f"train_corr_window_{corr_window}_seq_len_{seq_len}_corr_threshold_{corr_threshold}_lr_0.001"

    for epoch in range(40):
        loss, train_r2, y_true_t, y_pred_t = train(train_loader, model, optimizer, loss_fn)
        loss_test, val_r2, _, _ = evaluate(test_loader, model, loss_fn)
        logger.info(f"Epoch {epoch}, Train Loss: {loss:.4f}, Train R2: {train_r2:.4f}")
        logger.info(f"Validation Loss: {loss_test:.4f}, Validation R2: {val_r2:.4f}")
        loss_train.append(loss)
        loss_val.append(loss_test)
        r2_train.append(train_r2)
        r2_val.append(val_r2)
        scheduler.step(loss_test)
        # early_stopping(loss_test, model)
        # if early_stopping.early_stop:
        #     logger.info("Early stopping triggered")
        #     break

    plots(epoch, loss_train, loss_val, r2_val, r2_train, y_true_t, y_pred_t, name)
    # torch.save(model.state_dict(), f'huber2/model_{name}.pt')
    # logger.info(f"Model saved to huber2/model_{name}.pt")

if __name__ == "__main__":
    corr_list = [0.0,0.3, 0.5, 0.6, 0.7,0.8]
    windows_list = [5, 10, 15]
    seq_len_list = windows_list
    # seq_len_list = [5, 10]
    for corr in corr_list:
        for window in windows_list:
            # for seq_len in seq_len_list:
            runner(corr_window=window, seq_len=window, corr_threshold=corr)