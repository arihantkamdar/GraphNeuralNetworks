import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import r2_score

from model import MTGNN  # Assuming your model file is named model.py
from data_cooking import prepare_dataset, MTGNNDataset  # Your dataset code


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_trues = []


    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)  # x: [B, N, F, T], y: [B, N, pred_len]
        optimizer.zero_grad()
        out = model(x)  # [B, N, 1]
        out = out.squeeze(-1)  # [B, N]
        all_preds.append(out.cpu().detach())
        all_trues.append(y.squeeze(-1).detach().cpu())

        loss = criterion(out, y.squeeze(-1))  # [B, N]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    all_preds = torch.cat(all_preds, dim=0).numpy()  # [num_samples, num_nodes]
    all_trues = torch.cat(all_trues, dim=0).numpy()  # same shape
    plot_predictions(all_trues, all_preds, node_idx=0)

    r2 = r2_score(all_trues, all_preds) # if compute_r2 else None


    return total_loss / len(loader), r2

def plot_predictions(y_true, y_pred, node_idx=0, title="Predicted vs Actual",val = False,name = ''):
    """
    Plots predicted vs actual values over time for a given node.
    y_true, y_pred: [num_samples, num_nodes] (NumPy arrays or tensors)
    """
    parent_path = 'images'

    import numpy as np
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:, node_idx], label="Actual", linewidth=2)
    plt.plot(y_pred[:, node_idx], label="Predicted", linewidth=2, linestyle='--')
    plt.title(title + f" (Node {node_idx})")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    if val:
        plt.savefig("val_loop.png")
    else:
        plt.savefig("train_loop.png")
    plt.close()




def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(-1)
            loss = criterion(out, y.squeeze(-1))
            total_loss += loss.item()
            all_preds.append(out.cpu())
            all_trues.append(y.squeeze(-1).cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()  # [num_samples, num_nodes]
    all_trues = torch.cat(all_trues, dim=0).numpy()  # same shape
    plot_predictions(all_trues, all_preds, node_idx=0,val = True)
    adj1 = model.block1.graph_constructor().detach().cpu()
    adj2 = model.block2.graph_constructor().detach().cpu()
    visualize_adj_matrix(adj1, 1, name="block1")
    visualize_adj_matrix(adj2, 1, name="block2")

    r2 = r2_score(all_trues, all_preds) # if compute_r2 else None
    return total_loss / len(loader), r2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    seq_len = 10
    pred_len = 1
    batch_size = 8
    lr = 1e-4
    epochs = 150
    in_dim = 1
    hidden_dim = 64
    out_dim = 1
    gcn_hops = 3
    num_nodes = 12

    # Load data
    X_all, Y_all = prepare_dataset()
    X_train = X_all[:int(len(X_all)*0.7)]
    Y_train = Y_all[:int(len(Y_all)*0.7)]
    X_val = X_all[int(len(X_all)*0.7):int(len(X_all)*0.85)]
    Y_val = Y_all[int(len(X_all)*0.7):int(len(X_all)*0.85)]
    X_test = X_all[int(len(X_all)*0.85):]
    Y_test = Y_all[int(len(X_all)*0.85):]
    # dataset = MTGNNDataset(X_all, Y_all, seq_len=seq_len, pred_len=pred_len)

    # train_len = int(0.7 * len(dataset))
    # val_len = int(0.15 * len(dataset))
    # test_len = len(dataset) - train_len - val_len
    # train_set = dataset[:train_len]
    # val_set = dataset[train_len:train_len+val_len]
    # test_set = dataset[train_len+val_len:]
    # train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    train_set = MTGNNDataset(X_train,Y_train,seq_len=seq_len, pred_len=pred_len)
    test_set  = MTGNNDataset(X_test,Y_test,seq_len=seq_len, pred_len=pred_len)
    val_set = MTGNNDataset(X_val,Y_val,seq_len=seq_len, pred_len=pred_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Model
    model = MTGNN(num_nodes, in_dim, hidden_dim, out_dim, gcn_hops, device).to(device)
    print(model)

    criterion = nn.HuberLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # In main loop, after eval_epoch:
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_r2 = []
    val_r2 = []
    for epoch in range(1, epochs + 1):
        train_loss, r2_train = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, r2_val = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_r2.append(r2_train)
        val_losses.append(val_loss)


        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}")
        print(f"Epoch {epoch:02d} | Train r2: {r2_train:.8f} | Val r2: {r2_val:.8f}")
        scheduler.step(train_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_mtgcnn.pt")

    # Test Evaluation
    model.load_state_dict(torch.load("best_mtgcnn.pt"))
    plt.close()
    plt.plot(range(epochs),train_losses,label = "trainLoss")
    plt.savefig("trainloss.png")
    plt.close()
    plt.plot(range(epochs),train_r2,label = "trainr2")
    plt.savefig("trainr2.png")
    plt.close()
    test_loss,_ = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    # print(model.a)
    # for a,b in test_loader:
    #     b_out = model(a)
    #     print(b_out)
    #     print(b)
    #     break

import matplotlib.pyplot as plt

def visualize_adj_matrix(adj, epoch, name="block1"):
    plt.figure(figsize=(6, 5))
    plt.imshow(adj, cmap='viridis')
    plt.title(f"{name} Adjacency Matrix - Epoch {epoch}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{name}_adj_epoch_{epoch}.png")
    plt.close()



if __name__ == "__main__":
    main()