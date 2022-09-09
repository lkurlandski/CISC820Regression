"""
Viet and Luke
CISC-820: Quantitative Foundations
Project 1: Linear Feature Engineering
"""
# %%

import json

import numpy as np
import torch
import torch.nn as nn


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

def main():
    """Test different transformations and models and record predictions for test data.
    """

    test_data = np.loadtxt("testinputs.txt")
    
    history = np.load("./history.npz")
    
    best_fold = history["best_fold"]
    vl = history["mean_val_losses"]

    print(f"All Val Loss: {vl}")

    ## Load best model to predict on test
    model = LinearNet(test_data.shape[1], 16, 2, 7)
    model.load_state_dict(torch.load(f"./models/model_fold_{best_fold}.pth"))
    model.eval()

    with torch.no_grad():
        results = model(torch.from_numpy(test_data).float())

        np.savetxt("testoutputs_nn.txt", results.numpy(), fmt="%10.20f")

if __name__ == "__main__":
    main()
