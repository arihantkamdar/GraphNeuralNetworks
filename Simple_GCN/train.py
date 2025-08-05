import logging
import os
import random

import torch.nn
from sklearn.metrics import r2_score
from sympy.codegen.ast import continue_
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader

# from utils import EarlyStopping
from gcn import TemporalGCN
from construct_graphs import *
import matplotlib.pyplot as plt


logging.basicConfig(
    filename='training_logs.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mape_loss(y_pred, y_true, eps=1e-3):
    """
    Differentiable, stable MAPE loss for backpropagation.

    Args:
        y_pred (Tensor): Predicted values.
        y_true (Tensor): Ground truth values.
        eps (float): Epsilon to avoid division by zero.

    Returns:
        loss (Tensor): MAPE loss.
    """
    denom = torch.where(torch.abs(y_true) < eps, eps, torch.abs(y_true))
    loss = torch.abs((y_true - y_pred) / denom)
    return torch.mean(loss)
# Assuming your FinancialGraphSequenceDataset and STGCNModel are defined above


def train(graphs, model, optimizer, loss_fn):
    model.train()
    total_loss = 0
    y_true_all,y_pred_all = [],[]

    for data in graphs:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_true_all.append(data.y.detach())
        y_pred_all.append(out.detach())
    y_true_all = torch.cat(y_true_all).cpu().numpy().flatten()
    y_pred_all = torch.cat(y_pred_all).cpu().numpy().flatten()
    r2 = r2_score(y_true_all, y_pred_all)
    final_loss = total_loss / len(graphs)
    return final_loss, r2, y_true_all, y_pred_all


def evaluate(graphs, model, loss_fn):
    model.eval()
    total_loss = 0
    y_true_all,y_pred_all = [], []
    with torch.no_grad():
        for data in graphs:
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y)
            total_loss += loss.item()
            y_true_all.append(data.y.detach())
            y_pred_all.append(out.detach())
        y_true_all = torch.cat(y_true_all).cpu().numpy().flatten()
        y_pred_all = torch.cat(y_pred_all).cpu().numpy().flatten()
        r2 = r2_score(y_true_all, y_pred_all)
        final_loss = total_loss / len(graphs)

    return final_loss, r2


def plots(epoch, train_losses, val_losses, val_r2s, train_r2s, name,y_true_all, y_pred_all):
    y_true_all = y_true_all[:20]
    y_pred_all = y_pred_all[:20]
    plt.figure(figsize=(12, 5))
    epoch = len(train_losses)

    plt.subplot(1, 3, 1)
    plt.plot(range(epoch), train_losses, label=' Loss')
    # plt.plot(range(epoch), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    # plt.plot(range(epoch), val_r2s, label=' R2')
    plt.plot(range(epoch), train_r2s, label='R2')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.title('R2 Score over Epochs')
    plt.legend()
    plt.subplot(1, 3, 3)
    # plt.plot(range(epoch), val_r2s, label=' R2')
    plt.plot(range(len(y_true_all)), y_true_all, label='true')
    plt.plot(range(len(y_true_all)), y_pred_all, label='pred')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.title('Pred vs True Vals')
    plt.legend()

    plt.tight_layout()
    if not os.path.isdir('images'):
        os.makedirs('images')


    plt.savefig(f'images/{name}.png')
    plt.close()



def runner(corr_window, corr_threshold):
    df = pd.read_csv('/home/arihant/MSC/Data/formatted_data_clean_pct_change.csv')
    # cols = df.columns
    cols = ['BSE SENSEX Close', 'Crude Close',
       'Dow Jones Industrial Average Close', 'Euronext 100 Close',
       'FTSE 100 Index Close', 'Gold Close', 'NASDAQ Composite Close',
       'NYSE Composite Close', 'Nifty 50 Close', 'Nikkei 225 Close',
       'S&P 500 Index Close', 'SSE Composite Index Close', 'BSE SENSEX High',
       'Crude High', 'Dow Jones Industrial Average High', 'Euronext 100 High',
       'FTSE 100 Index High', 'Gold High', 'NASDAQ Composite High',
       'NYSE Composite High', 'Nifty 50 High', 'Nikkei 225 High',
       'S&P 500 Index High', 'SSE Composite Index High', 'BSE SENSEX Low',
       'Crude Low', 'Dow Jones Industrial Average Low', 'Euronext 100 Low',
       'FTSE 100 Index Low', 'Gold Low', 'NASDAQ Composite Low',
       'NYSE Composite Low', 'Nifty 50 Low', 'Nikkei 225 Low',
       'S&P 500 Index Low', 'SSE Composite Index Low', 'BSE SENSEX Open',
       'Crude Open', 'Dow Jones Industrial Average Open', 'Euronext 100 Open',
       'FTSE 100 Index Open', 'Gold Open', 'NASDAQ Composite Open',
       'NYSE Composite Open', 'Nifty 50 Open', 'Nikkei 225 Open',
       'S&P 500 Index Open', 'SSE Composite Index Open']
    df = df[cols]
    for col in cols:
        df[col] = StandardScaler().fit_transform(df[[col]])

    graphs = construct_simple_correlation_graphs_no_ss(df=df,corr_window=corr_window, corr_threshold=corr_threshold,self_loop=True)
    graphs = graphs[:500]
    random.seed(42)
    graphs_copy = graphs.copy()
    random.shuffle(graphs_copy)

    split = int(0.8 * len(graphs_copy))
    train_graphs = graphs_copy[:split]
    test_graphs = graphs_copy[split:]


    train_loader = DataLoader(train_graphs, batch_size=1)
    test_loader = DataLoader(test_graphs, batch_size=1)


    model = TemporalGCN(in_channels=4, hidden_channels=16, out_channels=1).to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.MSELoss()
    logging.info("************************************************************")
    logging.info(f"Corr_window: {corr_window} Corr_threshold: {corr_threshold} LR: 0.00001")
    logging.info("Graphs Constructed")
    loss_train, loss_val  =[], []
    r2_train, r2_val =[], []
    name = f"Corr_window: {corr_window} Corr_threshold: {corr_threshold} LR: 0.00001 All Scaled"
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    count = 0

    for epoch in range(70):
        loss,train_r2,y_true_all, y_pred_all = train(train_loader,model, optimizer, loss_fn)
        count+=1

        loss_test,val_r2 = evaluate(test_loader,model, loss_fn)
        logger.info(f"Epoch {epoch}, Train Loss: {loss:.4f} train r2 : {train_r2}")
        logger.info(f"Validation Loss: {loss_test:.4f}, Validation R2: {val_r2:.4f}")
        loss_train.append(loss)
        loss_val.append(loss_test)
        r2_train.append(train_r2)
        r2_val.append(val_r2)
        scheduler.step(loss_test)
        # if early_stopper(loss_test):
        #     logger.info(f"Early stopping at epoch {epoch}")
        #     break

    plots(epoch = count,train_losses=loss_train,train_r2s=r2_train,val_r2s=r2_val, val_losses=loss_val, name=name,
          y_true_all=y_true_all, y_pred_all=y_pred_all)


corr_list = [0,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
windows_list = [21,50,100,200]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Device: {device}")

for i in corr_list:
    for j in windows_list:
        # early_stopper = EarlyStopping(patience=10, min_delta=5e-5, verbose=True)
        runner(corr_threshold=i,corr_window=j)

