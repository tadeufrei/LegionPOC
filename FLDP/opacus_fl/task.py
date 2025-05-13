"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

# Force use of only the first CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ---------------------
# Model Definition
# ---------------------

class Net(nn.Module):
    """Simple MLP for binary classification using 77 input features."""

    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(77, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.hidden(x)


# ---------------------
# Model Parameter Utilities
# ---------------------

def get_model_weights(model: nn.Module) -> list[np.ndarray]:
    """Extract model weights as NumPy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_weights(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Set model weights from a list of NumPy arrays."""
    param_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
    model.load_state_dict(state_dict, strict=True)


# ---------------------
# Data Preprocessing
# ---------------------

def preprocess_can_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw CAN bus dataframe into binary-encoded features and labels.
    Assumes:
        - Column 1: CAN ID (hex string)
        - Column 2: Data Length
        - Column 3: Data bytes (space-separated hex string)
        - Column 4: Label ('R' or 'T')
    """
    df = df.copy()
    features = pd.DataFrame()

    # Convert CAN ID (col 1) to 12-bit binary
    features[0] = df[1].apply(lambda x: format(int(x, 16), "012b"))
    for bit in range(12):
        features[bit + 1] = features[0].apply(lambda b: int(b[bit]))
    features.drop(columns=[0], inplace=True)

    # Convert data length (col 2) to binary
    features[features.shape[1] + 1] = df[2].apply(lambda x: int(format(int(x), "b")))
    features.reset_index(drop=True, inplace=True)

    # Convert data bytes (col 3) to bits
    def expand_data_bytes(row):
        data_len = row[2]
        data_bytes = row[3].split()[:data_len]
        bitstring = ''.join(format(int(byte, 16), '08b') for byte in data_bytes)
        return [int(bit) for bit in bitstring]

    df[3] = df.apply(expand_data_bytes, axis=1)
    expanded = pd.DataFrame(df[3].tolist()).fillna(0).astype(int)
    features = pd.concat([features, expanded], axis=1)

    # Convert labels (col 4) to binary
    labels = df[4].map({"R": 0, "T": 1}).astype(np.int32)
    features[len(features.columns)] = labels
    features.columns = range(features.shape[1])

    return features


# ---------------------
# Differentially Private Training
# ---------------------

def train_with_privacy(
        model: nn.Module,
        train_loader: DataLoader,
        privacy_engine,
        optimizer: torch.optim.Optimizer,
        delta: float,
        device: torch.device,
        epochs: int = 10,
) -> float:
    """
    Train the model with sample-level differential privacy using Opacus.

    Returns:
        epsilon: Effective privacy budget after training.
    """
    model.to(device)
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=delta)
    return epsilon


# ---------------------
# Evaluation
# ---------------------

def evaluate_model(model: nn.Module, raw_test_df: pd.DataFrame, device: torch.device) -> tuple[float, float]:
    """
    Evaluate the model and print classification metrics.

    Returns:
        Dummy loss (0.0), and accuracy.
    """
    test_df = preprocess_can_to_binary(raw_test_df)
    X = test_df.iloc[:, :-1].values
    y = test_df.iloc[:, -1].values.astype(np.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        y_probs = torch.sigmoid(logits).cpu().numpy()

    y_true = y_tensor.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_probs > optimal_threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("[Evaluation]")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}\n")

    return 0.0, accuracy


# ---------------------
# Data Loading
# ---------------------

def load_train_data(folder_path: str) -> pd.DataFrame:
    """Load all CSV files from a folder into a single DataFrame."""
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    return pd.concat([pd.read_csv(f, header=None) for f in csv_files], ignore_index=True)


def load_test_partition(partition_id: int, test_file: str, num_clients: int) -> pd.DataFrame:
    """Return a single client's test data partition."""
    df = pd.read_csv(test_file, header=None)
    partitions = split_df_balanced_by_label(df, num_clients)
    return partitions[partition_id]


def split_df_balanced_by_label(df: pd.DataFrame, num_partitions: int) -> list[pd.DataFrame]:
    """Split a dataset into balanced partitions (equal 'R' and 'T' labels per client)."""
    partitions = [[] for _ in range(num_partitions)]

    df_R = df[df[4] == 'R'].reset_index(drop=True)
    df_T = df[df[4] == 'T'].reset_index(drop=True)

    per_client_R = len(df_R) // num_partitions
    per_client_T = len(df_T) // num_partitions

    for i in range(num_partitions):
        partitions[i].extend(df_R.iloc[i * per_client_R: (i + 1) * per_client_R].values.tolist())
        partitions[i].extend(df_T.iloc[i * per_client_T: (i + 1) * per_client_T].values.tolist())

    return [pd.DataFrame(p) for p in partitions]
