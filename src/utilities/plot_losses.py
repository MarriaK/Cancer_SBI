import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "text.usetex"       : False,
    "font.family"       : "serif",
    "font.serif"        : ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size"         : 10,
    "axes.labelsize"    : 9,
    "xtick.labelsize"   : 8,
    "ytick.labelsize"   : 8,
    "legend.fontsize"   : 8,
    "axes.titlesize"    : 9,
    "axes.linewidth"    : 0.8,
    "xtick.direction"   : "out",
    "ytick.direction"   : "out",
    "figure.dpi"        : 300,
})

# ── Checkpoint paths and display config ──────────────────────────────────────
MODELS = [
    {
        "name"      : "Plain NPE",
        "ckpt"      : "Plain_NPE/checkpoints/latest.pt",
        "train_key" : "training_loss",
        "val_key"   : "test_loss",
        "color"     : "#4A90D9",
    },
    {
        "name"      : "Base NPE",
        "ckpt"      : "Base_NPE/checkpoints/latest.pt",
        "train_key" : "training_loss",
        "val_key"   : "validation_loss",
        "color"     : "#E07B54",
    },
    {
        "name"      : "Set Transformer NPE",
        "ckpt"      : "SetTransformer_NPE/checkpoints/latest.pt",
        "train_key" : "training_loss",
        "val_key"   : "validation_loss",
        "color"     : "#2CA02C",
    },
]

out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

# ── Load histories ────────────────────────────────────────────────────────────
histories = []
for m in MODELS:
    if not os.path.exists(m["ckpt"]):
        print(f"[WARN] Checkpoint not found: {m['ckpt']} — skipping {m['name']}")
        continue
    ckpt = torch.load(m["ckpt"], map_location="cpu")
    history = ckpt.get("history", {})
    train_loss = history.get(m["train_key"], [])
    val_loss   = history.get(m["val_key"],   [])
    if not train_loss:
        print(f"[WARN] Empty history in {m['ckpt']} — skipping {m['name']}")
        continue
    histories.append({
        "name"       : m["name"],
        "color"      : m["color"],
        "train_loss" : train_loss,
        "val_loss"   : val_loss,
    })
    print(f"[{m['name']}] epochs={len(train_loss)}, "
          f"best_val={min(val_loss):.4f} @ epoch {int(np.argmin(val_loss)) + 1}")

if not histories:
    raise RuntimeError("No checkpoints found. Run training first.")

# ── Plot 1: Training loss (all models on one axes) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
fig.patch.set_facecolor("white")

for h in histories:
    epochs = range(1, len(h["train_loss"]) + 1)
    axes[0].plot(epochs, h["train_loss"], color=h["color"], linewidth=1.4,
                 label=h["name"])
    axes[1].plot(range(1, len(h["val_loss"]) + 1), h["val_loss"],
                 color=h["color"], linewidth=1.4, label=h["name"])

for ax, title in zip(axes, ["Training Loss", "Validation / Test Loss"]):
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.set_title(title)
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#AAAAAA")
        spine.set_linewidth(0.8)
    ax.legend(framealpha=0.9, facecolor="white", edgecolor="#AAAAAA")

fig.suptitle("Training and Validation Loss — Plain NPE vs Base NPE vs Set Transformer NPE",
             fontweight="bold", y=1.02)
plt.tight_layout()
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"loss_curves_combined.{ext}")
    fig.savefig(path, dpi=300 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
plt.close(fig)

# ── Plot 2: One subplot per model (train + val together) ──────────────────────
n = len(histories)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
if n == 1:
    axes = [axes]
fig.patch.set_facecolor("white")

for ax, h in zip(axes, histories):
    epochs_tr  = range(1, len(h["train_loss"]) + 1)
    epochs_val = range(1, len(h["val_loss"])   + 1)

    ax.plot(epochs_tr,  h["train_loss"], color=h["color"],      linewidth=1.4, label="Train")
    ax.plot(epochs_val, h["val_loss"],   color=h["color"],      linewidth=1.4,
            linestyle="--", label="Val / Test")

    # Mark best val epoch
    best_epoch = int(np.argmin(h["val_loss"])) + 1
    best_val   = min(h["val_loss"])
    ax.axvline(best_epoch, color="#CC2222", linewidth=0.9, linestyle=":",
               label=f"Best val epoch {best_epoch} ({best_val:.2f})")

    ax.set_title(h["name"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#AAAAAA")
        spine.set_linewidth(0.8)
    ax.legend(framealpha=0.9, facecolor="white", edgecolor="#AAAAAA")

fig.suptitle("Loss Curves per Model  (solid = train, dashed = val/test)",
             fontweight="bold", y=1.02)
plt.tight_layout()
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"loss_curves_per_model.{ext}")
    fig.savefig(path, dpi=300 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
plt.close(fig)
