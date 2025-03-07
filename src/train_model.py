import numpy as np
import pandas as pd
import json
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger

# Load configuration
config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

# Hyperparameters from config
train_ratio = config.get("train_ratio", 0.8)
random_seed = config.get("random_seed", 42)
svm_config = config.get("svm", {})
kernel = svm_config.get("kernel", "rbf")
C = svm_config.get("C", 1.0)
degree = svm_config.get("degree", 3)
gamma = svm_config.get("gamma", "scale")

# Load dataset
data_path = "data/Aggregation.txt"
data = pd.read_csv(data_path, sep="\t", skiprows=7, header=None, names=['X1', 'X2', 'y'])

# Extract features and labels
X = data[['X1', 'X2']].values
y = data['y'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_ratio, random_state=random_seed
)

# Train the SVM model
print(f"Training SVM with kernel={kernel}, C={C}, degree={degree}, gamma={gamma}...")
model = OneVsRestClassifier(SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, random_state=random_seed))
model.fit(X_train, y_train)

# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate accuracy and classification report
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
class_report = classification_report(y_test, test_predictions, output_dict=True)

# Save metrics
metrics = {
    "train_accuracy": train_accuracy,
    "test_accuracy": test_accuracy,
    "classification_report": class_report
}

metrics_file = "metrics.json"
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)
print("Metrics saved to", metrics_file)

# Save the trained model
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_file = os.path.join(model_dir, "svm_model.pkl")
joblib.dump(model, model_file)
print("Model saved to", model_file)

# Function to plot decision boundary
def plot_decision_boundary(X, y, clf, filename="static/decision_boundary.png"):
    zero_one_colourmap = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    matplotlib_axes_logger.setLevel('ERROR')

    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.1))

    plt.figure(figsize=(10, 5))
    plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(zero_one_colourmap[:len(np.unique(y_set))]))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=zero_one_colourmap[i], label=j)  # FIXED COLOR

    plt.title(f'SVM Decision Boundary (Kernel: {kernel})')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.savefig(filename)  # Save the plot as an image
    print(f"Decision boundary plot saved as {filename}")
    plt.close()  # Prevent displaying the image


# Generate and save decision boundary plot
plot_decision_boundary(X, y, model)
