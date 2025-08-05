import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import r2_score
from model import STGCN
import matplotlib.pyplot as plt

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


class STGCNTrainer:
    def __init__(self, num_nodes, in_features, hidden_dim, out_features, seq_len, batch_size=16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

        self.model = STGCN(num_nodes, in_features, hidden_dim, out_features, seq_len).to(self.device)
        self.criterion = nn.HuberLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.batch_size = batch_size
        self.num_nodes = num_nodes

    def train_epoch(self, train_loader,name):
        self.model.train()
        total_loss = 0
        true_vals, pred_vals = [], []

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(batch.x, batch.edge_index, batch.edge_attr)
            batch_y = batch.y.view(-1, self.num_nodes, 1).to(self.device)

            loss = self.criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            true_vals.append(batch_y.detach().cpu().numpy())
            pred_vals.append(output.detach().cpu().numpy())

        avg_loss = total_loss / len(train_loader)

        true_arr = np.concatenate(true_vals, axis=0).ravel()
        pred_arr = np.concatenate(pred_vals, axis=0).ravel()

        plt.figure()
        plt.plot(range(50), true_arr[:50], label="True")
        plt.plot(range(50), pred_arr[:50], label="Predicted")
        plt.legend()
        plt.title("Predictions vs True (Sample)")
        plt.savefig(f"{name}pred.png")
        plt.close()

        r2 = r2_score(true_arr, pred_arr)
        return avg_loss, r2

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        true_vals, pred_vals = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                output = self.model(batch.x, batch.edge_index, batch.edge_attr)
                batch_y = batch.y.view(-1, self.num_nodes, 1).to(self.device)

                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
                true_vals.append(batch_y.cpu().numpy())
                pred_vals.append(output.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        true_arr = np.concatenate(true_vals, axis=0).ravel()
        pred_arr = np.concatenate(pred_vals, axis=0).ravel()
        r2 = r2_score(true_arr, pred_arr)
        return avg_loss, r2

    def test(self, test_loader):
        self.model.eval()
        total_loss = 0
        predictions, actuals = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                output = self.model(batch.x, batch.edge_index, batch.edge_attr)
                batch_y = batch.y.view(-1, self.num_nodes, 1).to(self.device)

                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
                predictions.append(output.cpu().numpy())
                actuals.append(batch_y.cpu().numpy())

        avg_loss = total_loss / len(test_loader)

        true_arr = np.concatenate(actuals, axis=0).ravel()
        pred_arr = np.concatenate(predictions, axis=0).ravel()
        r2 = r2_score(true_arr, pred_arr)

        return avg_loss, r2, predictions, actuals

    def train_and_evaluate(self, graphs, name, train_split=0.7, val_split=0.1, epochs=35):
        train_size = int(len(graphs) * train_split)
        val_size = int(len(graphs) * val_split)

        train_data = graphs[:train_size]
        val_data = graphs[train_size:train_size + val_size]
        test_data = graphs[train_size + val_size:]

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        train_losses, val_losses = [], []
        train_r2s, val_r2s = [], []

        best_val_loss = float('inf')
        patience, counter = 20, 0

        for epoch in range(epochs):
            train_loss, train_r2 = self.train_epoch(train_loader,name)
            val_loss, val_r2 = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_r2s.append(train_r2)
            val_r2s.append(val_r2)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train R2: {train_r2:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}")

            self.scheduler.step(train_loss)

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     counter = 0
            #     torch.save(self.model.state_dict(), 'best_model.pt')
            # else:
            #     counter += 1
            #     if counter >= patience:
            #         print("Early stopping triggered.")
            #         break

        # Plotting
        epochs_range = range(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs_range, train_losses, label='Train Loss')
        # plt.plot(epochs_range, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.savefig(f"{name} loss.png")
        plt.close()

        plt.figure()
        plt.plot(epochs_range, train_r2s, label='Train R2')
        # plt.plot(epochs_range, val_r2s, label='Val R2')
        plt.xlabel('Epoch')
        plt.ylabel('R2 Score')
        plt.title('R2 Score vs Epoch')
        plt.legend()
        plt.savefig(f"{name}r2.png")
        plt.close()

        # Final test
        test_loss, test_r2, predictions, actuals = self.test(test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}")
        return test_loss, test_r2, predictions, actuals


if __name__ == "__main__":
    for i in [10,20,50,100,200]:
        for j in [0.0,0.3,0.5,0.7,0.9,1.0]:
            name = f"corr {j} window {i}"
            from data import construct_graphs
            assert torch.cuda.is_available(), "CUDA not available. Make sure your GPU is set up properly."
            graphs = construct_graphs(window=i, seq_l=i, corr_threshold=j)
            print(f"Sample graph shape: x={graphs[0].x.shape}, y={graphs[0].y.shape}")
            trainer = STGCNTrainer(
                num_nodes=12,
                in_features=4,
                hidden_dim=32,
                out_features=1,
                seq_len=i,
                batch_size=32
            )
            trainer.train_and_evaluate(graphs,name)
