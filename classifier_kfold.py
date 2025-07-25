import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import StratifiedKFold
import random
import classifier

# ---------- Dataset Class ----------
class LabeledSimulationDataset(Dataset):
    def __init__(self, root_dir, label, num_trials=25, use_bulk=False):
        self.root_dir = root_dir
        self.label = label
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
        trials = []
        for t in range(1, self.num_trials + 1):
            trial_dir = os.path.join(sim_path, str(t))
            file_name = "CNratios_bulk.npy" if self.use_bulk else "CNratios_largest.npy"
            file_path = os.path.join(trial_dir, file_name)
            if os.path.exists(file_path):
                trial_data = np.load(file_path)
                trials.append(torch.tensor(trial_data, dtype=torch.float32))
            else:
                nan_tensor = torch.full((44,), float('nan'))
                trials.append(nan_tensor)
        x_tensor = torch.stack(trials)
        return x_tensor, torch.tensor(self.label, dtype=torch.long)

# ---------- Data Preparation ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
high_dataset = LabeledSimulationDataset("numpy_data", label=1)
low_dataset  = LabeledSimulationDataset("numpy_data_low", label=0)
min_len = min(len(high_dataset), len(low_dataset))

balanced_high = Subset(high_dataset, random.sample(range(len(high_dataset)), min_len))
balanced_low  = Subset(low_dataset, random.sample(range(len(low_dataset)), min_len))

combined_dataset = ConcatDataset([balanced_high, balanced_low])
combined_labels = [1] * min_len + [0] * min_len

# ---------- K-Fold Setup ----------
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# ---------- Collect Fold Results ----------
all_val_f1 = []
all_val_precision = []
all_val_recall = []
all_val_loss = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(combined_labels)), combined_labels)):
    print(f"\n Fold {fold_idx + 1}/{k_folds}")

    train_dataset = Subset(combined_dataset, train_idx)
    val_dataset   = Subset(combined_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

    def count_labels(dataloader):
        all_labels = []
        for _, labels in dataloader:
            all_labels.extend(labels.numpy())
        return all_labels.count(0), all_labels.count(1)

    n0_train, n1_train = count_labels(train_loader)
    n0_val, n1_val = count_labels(val_loader)
    print(f"Train labels: 0 = {n0_train}, 1 = {n1_train}")
    print(f"Val labels:   0 = {n0_val}, 1 = {n1_val}")

    model = classifier.DeepSetClassifier(classifier.embedding_net).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    early_stopping = classifier.EarlyStopping(patience=50, delta=1)

    for epoch in range(20):
        loss_tr, f1_tr, precision_tr, recall_tr = classifier.train(model, train_loader, optimizer, loss_fn, device)
        loss_ts, f1_ts, precision_ts, recall_ts = classifier.evaluate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1:03d} | Train Loss = {loss_tr:.4f}, F1 = {f1_tr:.4f} | Val Loss = {loss_ts:.4f}, F1 = {f1_ts:.4f}")

        early_stopping(f1_ts, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save metrics from this fold
    all_val_f1.append(f1_ts)
    all_val_precision.append(precision_ts)
    all_val_recall.append(recall_ts)
    all_val_loss.append(loss_ts)

    # Optional: Save model for this fold
    torch.save(model.state_dict(), f"model_fold{fold_idx+1}.pth")
    print(f"Saved model for fold {fold_idx+1}")

# ---------- Overall Summary ----------
print("\n===== K-Fold Cross Validation Results =====")
print(f"Validation F1 Score:       {np.mean(all_val_f1):.4f} ± {np.std(all_val_f1):.4f}")
print(f"Validation Precision:      {np.mean(all_val_precision):.4f} ± {np.std(all_val_precision):.4f}")
print(f"Validation Recall:         {np.mean(all_val_recall):.4f} ± {np.std(all_val_recall):.4f}")
print(f"Validation Loss:           {np.mean(all_val_loss):.4f} ± {np.std(all_val_loss):.4f}")