import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from config import *
from model.autoencoder import Autoencoder
from utils.conversion_utils import ConversionUtils
from utils.visualization import Visualization

os.makedirs(DATASET_DIR_STL, exist_ok=True)
os.makedirs(DATASET_DIR_PLY, exist_ok=True)
os.makedirs(CHECKPOINT_PATH.rsplit("/", 1)[0], exist_ok=True)
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)


def binary_accuracy(output, target):
    return (output.round() == target).float().mean().item()

if not os.listdir(DATASET_DIR_PLY):
    print("\033[92m")
    print("══════════════════════════════════════════════════")
    print("          STARTING STL → PLY CONVERSION          ")
    print(f"  Files to convert: {len(ConversionUtils.list_files_in_directory(DATASET_DIR_STL)):<5}")
    print(f"  Point cloud density: {POINT_CLOUD_DENSITY:<6}")
    print(f"  Output directory: {DATASET_DIR_PLY}")
    print("══════════════════════════════════════════════════")
    print("\033[0m")
    for filename in tqdm(ConversionUtils.list_files_in_directory(DATASET_DIR_STL), desc="STL → PLY"):
        ConversionUtils.stl_to_ply(os.path.join(
            DATASET_DIR_STL, filename), POINT_CLOUD_DENSITY, DATASET_DIR_PLY)
else:
    print(f"Skipping STL → PLY conversion: {DATASET_DIR_PLY} already populated")

print("\033[92m")
print("══════════════════════════════════════════════════")
print("          STARTING PLY → BINVOX CONVERSION          ")
print(f"  Files to convert: {len(ConversionUtils.list_files_in_directory(DATASET_DIR_STL)):<5}")
print(f"  Point cloud density: {POINT_CLOUD_DENSITY:<6}")
print(f"  Output directory: {DATASET_DIR_PLY}")
print("══════════════════════════════════════════════════")
print("\033[0m")

dataset = []
for file in tqdm(ConversionUtils.list_files_in_directory(DATASET_DIR_PLY), desc="PLY → Voxels"):
    voxel = ConversionUtils.convert_to_binvox(
        os.path.join(DATASET_DIR_PLY, file), VOXEL_DIM)
    dataset.append(voxel)

dataset = np.expand_dims(np.array(dataset), axis=1).astype(np.float32)


split = int(0.8 * len(dataset))
train_data, test_data = dataset[:split], dataset[split:]

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_data)),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_data)),
    batch_size=BATCH_SIZE
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss()


PATIENCE = 5
MIN_DELTA = 0.0001
best_loss = float("inf")
patience_counter = 0
checkpoint_path = CHECKPOINT_PATH.rsplit(".", 1)[0] + ".pt"

print("\033[92m")
print("══════════════════════════════════════════════════")
print("              TRAINING AUTOENCODER                ")
print(f"  Epochs: {EPOCHS:<5} Batch Size: {BATCH_SIZE:<5} Device: {str(device):<6}")
print(f"  Train samples: {len(train_data):<5} Val samples: {len(test_data):<5}")
print("══════════════════════════════════════════════════")
print("\033[0m")

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for x_batch, y_batch in (pbar := tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x_batch.size(0)
        train_acc  += binary_accuracy(out, y_batch) * x_batch.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc":  f"{binary_accuracy(out, y_batch):.4f}"
        })

    train_loss /= len(train_loader.dataset)
    train_acc  /= len(train_loader.dataset)

    model.eval()
    val_loss, val_acc = 0.0, 0.0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            val_loss += criterion(out, y_batch).item() * x_batch.size(0)
            val_acc  += binary_accuracy(out, y_batch) * x_batch.size(0)

    val_loss /= len(test_loader.dataset)
    val_acc  /= len(test_loader.dataset)

    print(f"  loss: {train_loss:.4f}  acc: {train_acc:.4f}  │  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

    if train_loss < best_loss - MIN_DELTA:
        best_loss = train_loss
        patience_counter = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Checkpoint saved (loss: {best_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load(checkpoint_path, map_location=device))


model.eval()
reconstructed_batches, encoded_batches = [], []

with torch.no_grad():
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        reconstructed_batches.append(model(x_batch).cpu().numpy())
        encoded_batches.append(model.encode(x_batch).cpu().numpy())

reconstructed = np.concatenate(reconstructed_batches, axis=0)
encoded = np.concatenate(encoded_batches, axis=0)


torch.save(model.state_dict(),         os.path.join(SAVED_MODEL_DIR, "autoencoder.pt"))
torch.save(model.encoder.state_dict(), os.path.join(SAVED_MODEL_DIR, "encoder.pt"))

print("\033[92m")
print("══════════════════════════════════════════════════")
print("        TRAINING AND VALIDATION COMPLETE          ")
print("══════════════════════════════════════════════════")
print("\033[0m")
