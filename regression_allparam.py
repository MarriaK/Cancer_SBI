import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import random
from sklearn.model_selection import StratifiedShuffleSplit
import os
import torch.optim as optim
import torch.nn as nn

from net_builder import DeepSet
import NN_Utils

# Step 1: Load datasets with labels
lap_dataset = NN_Utils.LabeledSimulationDataset("numpy_data_laplace")
glow_dataset  = NN_Utils.LabeledSimulationDataset("numpy_data_low")
ghigh_dataset  = NN_Utils.LabeledSimulationDataset("numpy_data")

# Step 2: Balance datasets by truncating to min length
print(f"laplace dataset size: {len(lap_dataset)}")
print(f"gLow dataset size: {len(glow_dataset)}")
print(f"ghigh dataset size: {len(ghigh_dataset)}")

min_len = min(len(ghigh_dataset), len(glow_dataset))
min_len = min(len(lap_dataset), min_len)

balanced_lap = Subset(lap_dataset, random.sample(range(len(lap_dataset)), min_len))
balanced_glow  = Subset(glow_dataset, random.sample(range(len(glow_dataset)), min_len))
balanced_ghigh  = Subset(ghigh_dataset, random.sample(range(len(ghigh_dataset)), min_len))

# Step 3: Combine and shuffle
combined_dataset = ConcatDataset([balanced_lap, balanced_glow, balanced_ghigh ])
combined_labels = [2]*min_len + [0]*min_len + [1]*min_len  # Explicit labels for stratification

# Step 4: Stratified Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_idx, val_idx in sss.split(np.zeros(len(combined_labels)), combined_labels):
    train_dataset = Subset(combined_dataset, train_idx)
    val_dataset = Subset(combined_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)             

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NN_Utils.DeepSetRegression(NN_Utils.embedding_net, embedding_dim=64).to(device)  # Using DeepSetClassifier

optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

# Step 5: Training loop
num_epochs = 200
for epoch in range(num_epochs):
    loss_tr, r2_tr = NN_Utils.train(model, train_loader, optimizer, loss_fn, device)
    loss_ts, r2_ts = NN_Utils.evaluate(model, val_loader, loss_fn, device)
    
    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss = {loss_tr:.4f}, train R2 = {r2_tr:.4f} | "
        f"Val Loss = {loss_ts:.4f}, Val R2 = {r2_ts:.4f}"
    )