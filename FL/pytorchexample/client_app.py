"""pytorchexample: A Flower / PyTorch app."""

import os

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from pytorchexample.task import (
    Net,
    get_weights,
    set_weights,
    train,
    test,
    load_train_data,
    load_split_test_partition,
    preprocess_to_binary,
)
from torch.utils.data import DataLoader, TensorDataset

# Limit to the first CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowerClient(NumPyClient):
    """Federated Learning client using PyTorch and Flower."""

    def __init__(self, raw_train_df, test_loader, num_epochs: int, learning_rate: float):
        """
        Initialize the Flower client with model and training configuration.

        Args:
            raw_train_df: DataFrame with raw training data.
            test_loader: DataLoader with the client's test data.
            num_epochs: Number of local training epochs.
            learning_rate: Learning rate for training.
        """
        self.device = DEVICE
        self.model = Net().to(self.device)
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Preprocess training data and create DataLoader
        processed_train_df = preprocess_to_binary(raw_train_df)
        X = processed_train_df.iloc[:, :-1].values
        y = processed_train_df.iloc[:, -1].values.astype(np.float32)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    def fit(self, parameters, config):
        """
        Train the model with provided parameters.

        Args:
            parameters: Model weights received from the server.
            config: Optional training configuration.

        Returns:
            A tuple (updated_weights, training_size, metrics).
        """
        print(f"[Client] Using device: {self.device}")
        print(f"[Client] Model on: {next(self.model.parameters()).device}")

        set_weights(self.model, parameters)
        metrics = train(
            self.model,
            self.train_loader,
            self.test_loader,
            self.num_epochs,
            self.learning_rate,
            self.device,
        )
        return get_weights(self.model), len(self.train_loader), metrics

    def evaluate(self, parameters, config):
        """
        Evaluate the model with provided parameters.

        Args:
            parameters: Model weights received from the server.
            config: Optional evaluation configuration.

        Returns:
            A tuple (loss, test_size, metrics_dict).
        """
        print(f"[Client] Using device: {self.device}")
        print(f"[Client] Model on: {next(self.model.parameters()).device}")

        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader), {"accuracy": accuracy}


def build_client(context: Context) -> NumPyClient:
    """
    Create and return a configured Flower client instance.

    Args:
        context: Client context with configuration.

    Returns:
        A Flower NumPyClient instance.
    """
    partition_id = context.node_config["partition-id"]
    client_data_path = f"dataset/train/client{partition_id + 1}"

    # Load client's training data (as DataFrame)
    train_df = load_train_data(client_data_path)

    # Load the partitioned test set
    test_loader = load_split_test_partition(
        partition_id,
        test_file="dataset/test.csv",
        num_clients=4,  # Adjust based on the number of clients
    )

    return FlowerClient(
        raw_train_df=train_df,
        test_loader=test_loader,
        num_epochs=context.run_config["local-epochs"],
        learning_rate=context.run_config["learning-rate"],
    ).to_client()


# Define the Flower client app
app = ClientApp(build_client)
