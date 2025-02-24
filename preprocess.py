import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split

# Load dataset
train_path = "training.xlsx"  # Ensure this file is in your project folder
train_df = pd.read_excel(train_path)

# Identify dependent and independent variables
dependent_vars = ["A", "B", "C", "D", "E", "Y"]
spectral_columns = train_df.columns[7:]  # Assuming spectral data starts from the 8th column

# Extract features (X) and targets (Y)
X = train_df[spectral_columns].values  # Spectral data
Y = train_df[dependent_vars].values  # Target variables

# Step 1: Handle Missing Values (Fill with Mean)
X = np.nan_to_num(X, nan=np.nanmean(X))

# Step 2: Apply Savitzky-Golay Smoothing (Window=11, Poly=2)
X_smoothed = savgol_filter(X, window_length=11, polyorder=2, axis=1)

# Step 3: Standard Normal Variate (SNV) Transformation
def snv(input_data):
    """Applies Standard Normal Variate transformation."""
    return (input_data - np.mean(input_data, axis=1, keepdims=True)) / np.std(input_data, axis=1, keepdims=True)

X_snv = snv(X_smoothed)

# Step 4: Standardize Using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_snv)

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

# Step 5: Split into 80% Training / 20% Validation
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Save Preprocessed Data BEFORE Autoencoder (For CNN Training)
train_preprocessed = pd.DataFrame(X_train)
train_preprocessed[dependent_vars] = Y_train
train_preprocessed.to_csv("training_preprocessed.csv", index=False)

val_preprocessed = pd.DataFrame(X_val)
val_preprocessed[dependent_vars] = Y_val
val_preprocessed.to_csv("validation_preprocessed.csv", index=False)

print("✅ Preprocessing complete. Saved:")
print(" - training_preprocessed.csv (80% Training Data)")
print(" - validation_preprocessed.csv (20% Validation Data)")
print(" - scaler.pkl")
