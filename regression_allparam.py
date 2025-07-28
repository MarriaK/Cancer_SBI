import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
import random
from sklearn.model_selection import StratifiedShuffleSplit
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch.nn as nn
from torcheval.metrics import R2Score
from torcheval.metrics import R2Score
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F

from net_builder import DeepSet

class LabeledSimulationDataset(Dataset):
    def __init__(self, root_dir, num_trials=25, use_bulk=False):
        self.root_dir = root_dir
        self.num_trials = num_trials
        self.use_bulk = use_bulk
        self.sim_dirs = sorted(
            [d for d in os.listdir(root_dir) if d.startswith("sim")],
            key=lambda x: int(''.join(filter(str.isdigit, x)))
        )

    def __len__(self):
        return len(self.sim_dirs)

    def __getitem__(self, idx):
        sim_path = os.path.join(self.root_dir, self.sim_dirs[idx])
        parameter = np.load(f'{sim_path}/parameters.npy')
        parameter_mean = np.mean(parameter[2:])
        
        trials = []
        for t in range(1, self.num_trials + 1):
            trial_dir = os.path.join(sim_path, str(t))
            file_name = "CNratios_bulk.npy" if self.use_bulk else "CNratios_largest.npy"
            file_path = os.path.join(trial_dir, file_name)
            if os.path.exists(file_path):
                trial_data = np.load(file_path)
                trials.append(torch.tensor(trial_data, dtype=torch.float32))
            else:
                if trials:
                    nan_tensor = torch.full_like(trials[0], float('nan'))
                else:
                    nan_tensor = torch.full((44,), float('nan'))  # Default shape
                trials.append(nan_tensor)
        x_tensor = torch.stack(trials)  # shape: (num_trials, 44)
        return x_tensor, torch.tensor(parameter_mean), torch.tensor(parameter[2:])

# Step 1: Load datasets with labels
lap_dataset = LabeledSimulationDataset("numpy_data_laplace")
glow_dataset  = LabeledSimulationDataset("numpy_data_low")
ghigh_dataset  = LabeledSimulationDataset("numpy_data")

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
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False).                 


#embedding_net = DeepSet(hidden_dim_phi=64, hidden_dim_rho=64, output_dim=128)

embedding_net = DeepSet(
    hidden_dim_phi=128,   # increased from 64
    hidden_dim_rho=128,   # increased from 64
    output_dim=64,       # more expressive embedding
)
metric = R2Score()
        
class DeepSetRegression(nn.Module):
    def __init__(self, embedding_net, embedding_dim=64):
        super().__init__()
        self.embedding_net = embedding_net
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)
        self.fc2 = nn.Linear(64, 44)
    
    def forward(self, x):
        embedded = self.embedding_net(x)  # (B, embedding_dim)
        x = self.fc1(embedded)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # raw logits (no softmax yet)
        
        return logits  # return logits and last hidden layer

# ------------------------------
# Training Loop
# ------------------------------
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss,total_r2 = 0,0
    for x, _, label in dataloader:
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
        for x, _, labels in dataloader:
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = SimpleRegressor(input_dim=25 * 44, output_dim=44).to(device)
model = DeepSetRegression(embedding_net, embedding_dim=64).to(device)  # Using DeepSetClassifier

optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()