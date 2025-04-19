import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.amp import GradScaler, autocast
import numpy as np
import pickle
from models import GRUModel
from sklearn.preprocessing import MinMaxScaler
from run_bngl_avg import run_bionetgen_avg  # date April 11, 2025
from solve_ca_ode2 import solve_ca
import pandas as pd

# ----------------------------
#        DEVICE SETUP
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Enable cuDNN autotuner for faster performance on fixed input sizes
torch.backends.cudnn.benchmark = True


# ----------------------------
#       DATA LOADING
# ----------------------------
NUM_FEATURES = 1
NUM_TRAJECTORIES = 40000
all_input = NUM_TRAJECTORIES
input_arr = np.load("synthetic_input_parameters.npy")[:, :, 0].reshape(
    all_input, 2000, 1
)
output_arr = np.load("synthetic_output_parameters.npy")  # np.zeros((all_input, 5))

# ----------------------------
#       NORMALIZATION
# ----------------------------
# Reshape input: (N, T, F) → (N*T, F)
input_2d = input_arr.reshape(-1, NUM_FEATURES)
input_scaler = MinMaxScaler()
input_scaled = input_scaler.fit_transform(input_2d)
norm_in_arr = input_scaled.reshape(input_arr.shape)
# Output: shape (N, 5)
output_scaler = MinMaxScaler()
norm_out_arr = output_scaler.fit_transform(output_arr)
# Save scalers for use during prediction
with open("input_scaler.pkl", "wb") as f:
    pickle.dump(input_scaler, f)
with open("output_scaler.pkl", "wb") as f:
    pickle.dump(output_scaler, f)

# ----------------------------
#       DATASET PREP
# ----------------------------
input_tensor = torch.from_numpy(norm_in_arr).float()  # (N, 500, 3)
output_tensor = torch.from_numpy(norm_out_arr).float()  # (N, 500, 3)

# Create a single dataset with three outputs
dataset = TensorDataset(input_tensor, output_tensor)

# Train/Val Split
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)
val_size = dataset_size - train_size

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
#       MODEL SETUP
# ----------------------------
input_dim = NUM_FEATURES
hidden_dim = 128
output_dim = 5
num_layers = 1  # number of GRU layers
dropout = 0.0  # dropout probability is the same for all layers regressors
bidirectional = False

model = GRUModel(
    input_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout
).to(device)
# Optional: Compile model (requires PyTorch 2.0+)
model = torch.compile(model)

# We will have a combined loss for trajectory and parameter
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scaler = GradScaler(device="cuda")


# ----------------------------
#   Predict on external data during training
# ----------------------------
fin_data = pd.read_csv("M_46L_50F.dat", sep="\t", comment="#", header=None)
fin_data_y = fin_data.iloc[:, 1].to_numpy().reshape(-1, 1)  # shape: (2000, 1)
# Scale using the input scaler from training
test_data = input_scaler.transform(fin_data_y).reshape(1, fin_data_y.shape[0], 1)
test_tensor = torch.from_numpy(test_data).float().to(device)

data = pd.read_csv(
    "expt_ca_data/46L_50F_100k.dat", sep="\t", comment="#", header=None
)  # ZAP only
ca_signal_data = data.iloc[:, 1].values

# ----------------------------
#      TRAINING LOOP
# ----------------------------
num_epochs = 50
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda"):
            # Forward pass
            pred_output = model(batch_X)

            # Calculate combined loss
            loss = criterion(pred_output, batch_y)

        # Backward and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)

    # Validation
    model.eval()
    with torch.no_grad():
        ######################################
        ########## TRAJECTORY LOSS ###########
        ######################################
        pred_on_test_tensor = model(test_tensor).cpu().numpy()
        estimates = output_scaler.inverse_transform(pred_on_test_tensor)
        kon, koff, C1, C2, g = estimates[0]

        time, mean_pZAP = run_bionetgen_avg(3, 1, kon, koff)
        tnew, ca_pred = solve_ca(time, mean_pZAP, C1, C2, g)  # solving the ODEs

        denom = np.max(ca_signal_data)
        diff = (ca_pred[26:] - ca_signal_data) / denom
        diff = np.nan_to_num(diff, nan=0.0, posinf=1e3, neginf=-1e3)
        diff = np.clip(diff, -1e3, 1e3)
        running_val_loss = np.sum(diff**2)

        for val_X, val_y in val_loader:
            val_X = val_X.to(device)
            val_y = val_y.to(device)

            with autocast(device_type="cuda"):

                pred_output_val = model(val_X)
                val_loss = criterion(pred_output_val, val_y)
                running_val_loss += val_loss.item()

    val_loss = running_val_loss / len(val_loader)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "intrinsic_Ca_model2.pth")
        output_df = pd.DataFrame(
            {"time": tnew[26:], "ca_data": ca_signal_data, "ca_pred": ca_pred[26:]}
        )
        output_df.to_csv("best_prediction_ca_signal.csv", sep=",", index=False)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f} -- Model saved"
        )
    else:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )
