import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# ✅ Load Preprocessed Validation Data
val_data = pd.read_csv("validation_preprocessed.csv")

# Extract Features & Targets
dependent_vars = ["A", "B", "C", "D", "E", "Y"]
X_val = val_data.drop(columns=dependent_vars).values
Y_val = val_data[dependent_vars].values

# ✅ Load CNN Model
class SpectralCNN(nn.Module):
    def __init__(self, input_dim):
        super(SpectralCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding="same", dilation=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding="same")
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding="same")
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding="same")

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Compute FC Input Size
        self._to_linear = None
        self._calculate_flattened_size(input_dim)

        self.fc = nn.Linear(self._to_linear, 32)
        self.output = nn.Linear(32, 6)

    def _calculate_flattened_size(self, input_dim):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.pool(torch.relu(self.conv1(dummy_input)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = self.pool(torch.relu(self.conv4(x)))
            self._to_linear = x.view(1, -1).shape[1]

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


# ✅ Load CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpectralCNN(input_dim=X_val.shape[1]).to(device)
model.load_state_dict(torch.load("spectral_cnn.pth"))
model.eval()

# Convert Validation Data to Tensor
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)

# ✅ Extract CNN Features
with torch.no_grad():
    cnn_features_val = model.feature_extractor(X_val_tensor).cpu().numpy()

print("✅ CNN features extracted correctly.")

# ✅ Load SVM & KNN Models
svm_models = joblib.load("svm_models.pkl")
knn_models = joblib.load("knn_models.pkl")

# ✅ Generate Predictions
svm_predictions = np.array([svm_models[var].predict(cnn_features_val) for var in dependent_vars]).T
knn_predictions = np.array([knn_models[var].predict(cnn_features_val) for var in dependent_vars]).T

# ✅ Combine CNN, SVM, and KNN Outputs
final_predictions = (svm_predictions + knn_predictions) / 2

# ✅ Compute MAE for Validation
mae_scores = {var: mean_absolute_error(Y_val[:, i], final_predictions[:, i]) for i, var in enumerate(dependent_vars)}

# ✅ Compute Percentage Error for Validation
percentage_errors = {
    var: np.mean(np.abs((Y_val[:, i] - final_predictions[:, i]) / Y_val[:, i])) * 100
    for i, var in enumerate(dependent_vars)
}

# ✅ Save Predictions
predictions_df = pd.DataFrame(final_predictions, columns=dependent_vars)
predictions_df.to_csv("final_predicted_values.csv", index=False)

# ✅ Save Error Metrics
errors_df = pd.DataFrame([mae_scores, percentage_errors], index=["MAE", "Percentage Error"])
errors_df.to_csv("error_metrics.csv")

print("Predictions & error metrics saved.")
print("Mean Absolute Errors:", mae_scores)
print("Percentage Errors (%):", percentage_errors)

# ✅ Show Actual vs. Predicted Values
results = []
print("\n--- Actual vs. Predicted Values ---")
print(f"{'Variable':<10} {'Actual':<20} {'Predicted':<20} {'MAE':<10}")

for i, var in enumerate(dependent_vars):
    actual_values = Y_val[:, i]
    predicted_values = final_predictions[:, i]
    mae = mae_scores[var]

    for actual, predicted in zip(actual_values, predicted_values):
        results.append([var, actual, predicted])

    print(f"{var:<10} {actual_values[0]:<20.4f} {predicted_values[0]:<20.4f} {mae:.4f}")

# ✅ Save Actual vs. Predicted Values to CSV
results_df = pd.DataFrame(results, columns=["Variable", "Actual Value", "Predicted Value"])
results_df.to_csv("actual_vs_predicted.csv", index=False)

print("\n✅ Actual vs. Predicted values saved to 'actual_vs_predicted.csv'.")

# ✅ Visualization - CNN Feature Extraction
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(cnn_features_val[i], label=f"Sample {i+1}")
plt.xlabel("Feature Index")
plt.ylabel("Feature Value")
plt.title("CNN Extracted Features for Validation Data")
plt.legend()
plt.savefig("cnn_features_plot.png")
plt.show()

# ✅ Visualization - Prediction Errors
plt.figure(figsize=(10, 6))
plt.bar(percentage_errors.keys(), percentage_errors.values(), color='blue')
plt.xlabel("Dependent Variable")
plt.ylabel("Percentage Error (%)")
plt.title("Prediction Errors (Percentage) per Target Variable")
plt.savefig("prediction_errors_plot.png")
plt.show()

print("✅ Visualization complete. Check 'cnn_features_plot.png' & 'prediction_errors_plot.png'.")
