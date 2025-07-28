import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
import random
from sklearn.model_selection import StratifiedShuffleSplit
import torch.optim as optim
import torch.nn as nn
import NN_Utils
from torcheval.metrics import R2Score

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
#model = SimpleRegressor(input_dim=25 * 44, output_dim=44).to(device)
model = NN_Utils.DeepSetRegressionone(NN_Utils.embedding_net, embedding_dim=64).to(device)  # Using DeepSet from net_builder

optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()
metric = R2Score()

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss,total_r2 = 0,0
    for x,label, _ in dataloader:
        x, label = x.to(device), label.to(device)

        pred = model(x)  # (B, 44, 2)
        loss = loss_fn(pred, label)  # CrossEntropy expects (N, C) and (N,)
        metric.update(pred, label)
        total_r2 += metric.compute().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    #all_preds = torch.cat(all_preds).view(-1).numpy()
    #all_labels = torch.cat(all_labels).view(-1).numpy()

    #f1 = f1_score(all_labels, all_preds, average='macro')
    #precision = precision_score(all_labels, all_preds, average='macro')
    #recall = recall_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    avg_r2 = total_r2 / len(dataloader)
    return avg_loss, avg_r2



def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_loss,total_r2 = 0,0
    with torch.no_grad():
        for x, labels, _ in dataloader:
            x, labels = x.to(device), labels.to(device)
            pred = model(x)  # (B, 44, 3)
            metric.update(pred, labels)
            total_r2 += metric.compute()

            loss = loss_fn(pred, labels)  # CrossEntropy expects (N, C) and (N,)
            metric.update(pred, labels)
            total_loss += loss.item()


    avg_loss = total_loss / len(dataloader)
    avg_r2 = total_r2 / len(dataloader)
    return avg_loss, avg_r2   

# Step 5: Training loop
num_epochs = 200
for epoch in range(num_epochs):
    loss_tr, r2_tr = train(model, train_loader, optimizer, loss_fn, device)
    loss_ts, r2_ts = evaluate(model, val_loader, loss_fn, device)
    
    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss = {loss_tr:.4f}, R2 = {r2_tr:.4f} | "
        f"Val Loss = {loss_ts:.4f}, R2 = {r2_ts:.4f}"
    )
    
    