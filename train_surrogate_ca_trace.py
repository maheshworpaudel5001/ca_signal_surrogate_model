import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pickle
from models import SurrogateCaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Enable cuDNN autotuner for faster performance on fixed input sizes
torch.backends.cudnn.benchmark = True

# ----------------------------
# Load data
# ----------------------------
input_arr = np.load("synthetic_output_parameters.npy")
output_arr = np.load("synthetic_input_parameters.npy")[:, :, 0]

# Normalize inputs and outputs
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(input_arr)
Y_scaled = y_scaler.fit_transform(output_arr)

with open("surrogate_input_scaler.pkl", "wb") as f:
    pickle.dump(x_scaler, f)
with open("surrogate_output_scaler.pkl", "wb") as f:
    pickle.dump(y_scaler, f)

# Convert to torch tensors
X_tensor = torch.from_numpy(X_scaled).float()
Y_tensor = torch.from_numpy(Y_scaled).float()

# Dataset and dataloaders
dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 1024
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=8,
    drop_last=True,
)

# ----------------------------
# Model
# ----------------------------
model = SurrogateCaModel(input_dim=5, hidden_dim=64, output_dim=output_arr.shape[1]).to(
    device
)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# ----------------------------
# Training loop
# ----------------------------
num_epochs = 50
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, Y_val in val_loader:
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            pred_val = model(X_val)
            loss = criterion(pred_val, Y_val)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "surrogate_model.pth")
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} -- Model saved"
        )
    else:
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )
