import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Preprocessed Data
train_data = pd.read_csv("training_preprocessed.csv")
val_data = pd.read_csv("validation_preprocessed.csv")

# Extract Features & Targets
dependent_vars = ["A", "B", "C", "D", "E", "Y"]
X_train = train_data.drop(columns=dependent_vars).values
Y_train = train_data[dependent_vars].values
X_val = val_data.drop(columns=dependent_vars).values
Y_val = val_data[dependent_vars].values

# === ✅ ERROR CHECK: Validate Data Shape Before Training ===
if X_train.shape[1] != X_val.shape[1]:
    raise ValueError(f"❌ Feature size mismatch: X_train has {X_train.shape[1]} columns, but X_val has {X_val.shape[1]}.")

if Y_train.shape[1] != len(dependent_vars):
    raise ValueError(f"❌ Target variable mismatch: Expected {len(dependent_vars)}, but found {Y_train.shape[1]}.")

if np.any(np.isnan(X_train)) or np.any(np.isnan(X_val)) or np.any(np.isnan(Y_train)) or np.any(np.isnan(Y_val)):
    raise ValueError("❌ NaN values detected in the dataset! Check preprocessing.")

print("✅ Data shape check passed.")

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)

# Create DataLoader
batch_size = 8
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define CNN Model
class SpectralCNN(nn.Module):
    def __init__(self, input_dim):
        super(SpectralCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding="same", dilation=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding="same")
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding="same")
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding="same")

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Compute Flattened Size Dynamically
        self._to_linear = None
        self._calculate_flattened_size(input_dim)

        self.fc = nn.Linear(self._to_linear, 32)
        self.output = nn.Linear(32, 6)

    def _calculate_flattened_size(self, input_dim):
        """Pass a dummy input through CNN layers to determine FC input size."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim).to(device)
            x = self.pool(torch.relu(self.conv1(dummy_input)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = self.pool(torch.relu(self.conv4(x)))
            self._to_linear = x.view(1, -1).shape[1]  # Flattened size

    def feature_extractor(self, x):
        """Extract CNN Features Before Final Output Layer"""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))

        x = x.view(x.shape[0], -1)  # Flatten dynamically
        x = torch.relu(self.fc(x))
        return x  # ✅ Returns (batch_size, 32) features

    def forward(self, x):
        features = self.feature_extractor(x)  # ✅ Extract features first
        x = self.dropout(features)
        return self.output(x)  # ✅ Return final output (batch_size, 6)


# Instantiate CNN Model
model = SpectralCNN(input_dim=X_train.shape[1]).to(device)

# === ✅ ERROR CHECK: Validate CNN Model ===
# ✅ Ensure CNN model works correctly
try:
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, X_train.shape[1]).to(device)
        dummy_output = model(dummy_input)
        assert dummy_output.shape[1] == 6, f"❌ CNN output shape mismatch: Expected 6, got {dummy_output.shape[1]}"
    print("✅ CNN model structure is correct.")
except Exception as e:
    raise ValueError(f"❌ Error in CNN model structure: {e}")

# ✅ Define Loss Function & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.33)

# ✅ Train CNN
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)  # ✅ CNN Final Output
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    # ✅ Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            val_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

print("✅ Training loop completed.")

# ✅ Save Trained Model
torch.save(model.state_dict(), "spectral_cnn.pth")
print("✅ Model saved successfully.")

# ✅ Extract Features for SVM/KNN
model.eval()
with torch.no_grad():
    cnn_features_train = model.feature_extractor(X_train_tensor).cpu().numpy()
    cnn_features_val = model.feature_extractor(X_val_tensor).cpu().numpy()

print("✅ CNN features extracted correctly.")

# ✅ Train SVM & KNN
svm_models = {var: SVR() for var in dependent_vars}
knn_models = {var: KNeighborsRegressor(n_neighbors=5) for var in dependent_vars}

for i, var in enumerate(dependent_vars):
    svm_models[var].fit(cnn_features_train, Y_train[:, i])
    knn_models[var].fit(cnn_features_train, Y_train[:, i])

# ✅ Save SVM & KNN Models
joblib.dump(svm_models, "svm_models.pkl")
joblib.dump(knn_models, "knn_models.pkl")

print("✅ Training complete. Models saved.")
