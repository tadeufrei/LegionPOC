"""pytorchexample: A Flower / PyTorch app."""
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

# Force the use of the first CUDA device (if available)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ---------- Model ----------

class Net(nn.Module):
    """Simple feedforward neural network for binary classification."""

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


# ---------- Model Parameters Handling ----------

def get_weights(model: nn.Module) -> list[np.ndarray]:
    """Extract weights from a PyTorch model as NumPy arrays."""
    return [param.cpu().numpy() for param in model.state_dict().values()]


def set_weights(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Set weights for a PyTorch model from NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# ---------- Data Loading and Preprocessing ----------

def preprocess_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw CAN bus DataFrame into binary feature vectors and labels."""
    df = df.copy()
    features = pd.DataFrame()

    # Encode the second column (CAN ID) into binary
    features[0] = df[1].apply(lambda x: format(int(x, 16), "012b"))
    for bit_pos in range(12):
        features[bit_pos + 1] = features[0].apply(lambda x: int(x[bit_pos]))
    features.drop(columns=[0], inplace=True)

    # Encode the third column (data length) into binary
    features[features.shape[1] + 1] = df[2].apply(lambda x: int(format(int(x), "b")))
    features.reset_index(drop=True, inplace=True)

    # Process data bytes into individual bits
    def convert_data(row):
        data_length = row[2]
        data_values = row[3].split()[:data_length]
        binary_values = [format(int(byte, 16), "08b") for byte in data_values]
        return [int(bit) for byte in binary_values for bit in byte]

    df[3] = df.apply(convert_data, axis=1)
    expanded_data = pd.DataFrame(df[3].tolist()).fillna(0).astype(int)

    # Extract and convert labels
    label_column = df[4].map({"R": 0, "T": 1}).astype(np.int32)

    # Combine all features
    features = pd.concat([features, expanded_data], axis=1)
    features[len(features.columns)] = label_column
    features.columns = range(features.shape[1])

    return features


def load_train_data(folder_path: str) -> pd.DataFrame:
    """Load training data from all CSV files in a folder."""
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    print(f"[Data] Loading training files: {csv_files}")
    return pd.concat([pd.read_csv(f, header=None) for f in csv_files], ignore_index=True)


def load_split_test_partition(partition_id: int, test_file: str, num_clients: int) -> pd.DataFrame:
    """Load and return the partitioned test data for a specific client."""
    df = pd.read_csv(test_file, header=None)
    split_dfs = split_dataframe_balanced(df, num_clients)
    return split_dfs[partition_id]


def split_dataframe_balanced(df: pd.DataFrame, num_splits: int) -> list[pd.DataFrame]:
    """Split a DataFrame into balanced partitions by label ('R', 'T')."""
    splits = [[] for _ in range(num_splits)]
    rows_R = df[df[4] == 'R'].reset_index(drop=True)
    rows_T = df[df[4] == 'T'].reset_index(drop=True)

    count_R = len(rows_R)
    count_T = len(rows_T)
    per_client_R = count_R // num_splits
    per_client_T = count_T // num_splits

    for i in range(num_splits):
        splits[i].extend(rows_R.iloc[i * per_client_R: (i + 1) * per_client_R].values.tolist())
        splits[i].extend(rows_T.iloc[i * per_client_T: (i + 1) * per_client_T].values.tolist())

    return [pd.DataFrame(split) for split in splits]


# ---------- Training and Evaluation ----------

def train(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_data: pd.DataFrame,
        num_epochs: int,
        learning_rate: float,
        device: torch.device,
) -> dict:
    """Train the model locally."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"[Training] Epoch {epoch}, Loss: {loss.item():.4f}")

    val_loss, val_accuracy = test(model, test_data, device)
    return {"val_loss": val_loss, "val_accuracy": val_accuracy}


def test(
        model: nn.Module,
        raw_test_data: pd.DataFrame,
        device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on test data."""
    model.to(device)
    model.eval()

    test_df = preprocess_to_binary(raw_test_data)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.astype(np.float32)

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    y_true = y_tensor.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    preds_bin = (probs > optimal_threshold).astype(int)

    accuracy = accuracy_score(y_true, preds_bin)
    f1 = f1_score(y_true, preds_bin, zero_division=1)
    recall = recall_score(y_true, preds_bin, zero_division=1)
    conf_matrix = confusion_matrix(y_true, preds_bin)

    print("[Client Evaluation]")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}\n")

    return 0.0, accuracy  # Placeholder loss
