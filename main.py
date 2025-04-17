import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.amp import GradScaler, autocast
import numpy as np
import pickle
from models import GRUModel

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
from sklearn.preprocessing import MinMaxScaler

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
# input_arr = np.zeros((all_input, 2000, NUM_FEATURES))
# output_arr = np.zeros((all_input, 5))

# directory_path = "/home/gddaslab/share/MAITREYA_INDRANI/generate_Ca_training_data/training_data_mean/data_training_2"

# i = 0
# for filename in os.listdir(directory_path):
#     if os.path.isfile(os.path.join(directory_path, filename)):
#         output_arr[i] = np.loadtxt(directory_path + "/" + filename, max_rows=1)
#         input_arr[i] = np.loadtxt(
#             directory_path + "/" + filename, skiprows=1, usecols=1
#         ).reshape(2000, NUM_FEATURES)
#         i = i + 1
#         if i >= all_input:
#             break


# # ----------------------------
# #       NORMALIZATION
# # ----------------------------
# # 1) Normalize the input trajectories (input_dim=3)
# in_min_vals = input_arr.min(axis=(0, 1))  # shape (3,)
# in_max_vals = input_arr.max(axis=(0, 1))  # shape (3,)

# norm_in_arr = (input_arr - in_min_vals) / (in_max_vals - in_min_vals + 1e-9)

# # 2) Normalize the actual output trajectories (output_dim=3)
# out_min_vals = output_arr.min(axis=0)  # shape (3,)
# out_max_vals = output_arr.max(axis=0)  # shape (3,)

# norm_out_arr = (output_arr - out_min_vals) / (out_max_vals - out_min_vals + 1e-9)

# # Save normalization values for later usage
# with open("in_min_vals_intrinsic_Ca.pkl", "wb") as f:
#     pickle.dump(in_min_vals, f)
# with open("in_max_vals_intrinsic_Ca.pkl", "wb") as f:
#     pickle.dump(in_max_vals, f)

# with open("out_min_vals_intrinsic_Ca.pkl", "wb") as f:
#     pickle.dump(out_min_vals, f)
# with open("out_max_vals_intrinsic_Ca.pkl", "wb") as f:
#     pickle.dump(out_max_vals, f)


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
#      TRAINING LOOP
# ----------------------------
num_epochs = 200
best_val_loss = float("inf")

from tqdm import tqdm

for epoch in trange(num_epochs, total=num_epochs, leave=True):
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
    running_val_loss = 0.0
    with torch.no_grad():
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
        torch.save(model.state_dict(), "intrinsic_Ca_model.pth")
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f} -- Model saved"
        )
    else:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )
