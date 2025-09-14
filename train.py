import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim
from model import EEGLSTM

class EEGDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 20) = (N, 5*4), y: (N,)
        self.X = X.reshape(-1, 5, 4).astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def train():
    # load precomputed features
    X = np.load('X.npy')
    y = np.load('y.npy')
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    train_ds = EEGDataset(X_train, y_train)
    val_ds   = EEGDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGLSTM(bidirectional=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred==yb).sum().item()
        acc = correct / len(val_ds)

        print(f"Epoch {epoch:02d}  Loss {avg_loss:.4f}  Val Acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_eeg_lstm.pth')

    print("Best val accuracy:", best_acc)

if __name__=='__main__':
    train()