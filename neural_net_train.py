"""
Viet and Luke
CISC-820: Quantitative Foundations
Project 1: Linear Feature Engineering
"""
# %%

import json
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class LinearNet(torch.nn.Module):
    def __init__(self, input_features, starting_features: int, expand_ratio: int, n_expand: int) -> None:
        super().__init__()

        seq = []
        seq.append(nn.Linear(input_features, starting_features))
        seq.append(nn.ReLU())
        
        prev_feature = starting_features

        for i in range(n_expand):
            next_features = int(prev_feature * expand_ratio)
            seq.append(nn.Linear(prev_feature, next_features))
            seq.append(nn.BatchNorm1d(next_features))
            seq.append(nn.ReLU())
            prev_feature = next_features

        seq.append(nn.Linear(prev_feature, 1))

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

class NpDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
            
        if self.transform:
            pass
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def cross_validation(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:

    folds = []

    n = len(y)
    for k_i in range(k):
        start = k_i * n // k
        end = (k_i + 1) * n // k
        X_train, X_val = np.concatenate((X[0:start], X[end:n])), X[start:end]
        y_train, y_val = np.concatenate((y[0:start], y[end:n])), y[start:end]
        folds.append((X_train, X_val, y_train, y_val))
    return folds

def train_and_save_model(X_train, X_val, y_train, y_val, fold):

    n_data, n_features = X_train.shape

    train_dataset = NpDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True, drop_last=True)

    val_dataset = NpDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset,batch_size=32, shuffle=True, drop_last=True)
    val_it = iter(val_loader)


    model = LinearNet(n_features, 16, 2, 7)
    # res = model(torch.from_numpy(X).float())
    model.train()
    model_path = f"./models/model_fold_{fold}.pth"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 10
    best_mean_val_loss = 1e8
    best_train_loss = 1e8
    for i in range(n_epochs):
        print(f"------------ Epoch {i + 1} ---------")
        for j, [batch_x, batch_y] in enumerate(train_loader):
            output = model(batch_x)
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            
            # Compute loss and back prop
            loss = torch.nn.functional.mse_loss(output, batch_y)
            print (f"[Fold {fold}, Epoch {i + 1}, Step {j + 1}] Training Loss: {loss}")
            loss.backward()
            
            # Adjust learning weights
            optimizer.step()
        
        with torch.no_grad():
            val_out = model(torch.from_numpy(X_val).float())
            val_label = torch.from_numpy(y_val).float()
            val_loss = torch.nn.functional.mse_loss(val_out, val_label)
            print(f"[Epoch {i + 1}] Validation loss: {val_loss}")
            # val_losses.append(val_loss.item())

            out = model(torch.from_numpy(X_train).float())
            label = torch.from_numpy(y_train).float()
            loss = torch.nn.functional.mse_loss(out, label)

        ## Save model
        if val_loss < best_mean_val_loss:
            best_mean_val_loss = val_loss
            best_train_loss = loss
            torch.save(model.state_dict(), model_path)

    return best_mean_val_loss, best_train_loss

def main():
    """Test different transformations and models and record predictions for test data.
    """
    os.makedirs("./models", exist_ok=True)

    test_data = np.loadtxt("testinputs.txt")
    train_data = np.loadtxt("traindata.txt")
    X, y = train_data[:, 0:8], train_data[:, 8:]
    
    fold_data = cross_validation(X, y, 10)

    best_loss = 1e8

    best_fold = -1
    mean_val_losses = []
    mean_losses = []
    for fold, [X_train, X_val, y_train, y_val] in enumerate(fold_data):
        mean_val_loss, mean_loss = train_and_save_model(X_train, X_val, y_train, y_val, fold)
        mean_val_losses.append(mean_val_loss)
        mean_losses.append(mean_loss)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_fold = fold


    np.savez("history.npz", best_fold=best_fold, mean_val_losses=mean_val_losses, mean_losses=mean_losses)

    print("Done!")

if __name__ == "__main__":
    main()
