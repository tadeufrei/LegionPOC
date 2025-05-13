"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import os
import warnings

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from opacus import PrivacyEngine
from opacus_fl.task import (
    Net,
    get_model_weights,
    set_model_weights,
    train_with_privacy,
    evaluate_model,
    load_train_data,
    load_test_partition,
    preprocess_can_to_binary,
)
from torch.utils.data import DataLoader, TensorDataset

# Suppress Opacus warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)

# Restrict to the first CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DifferentialPrivacyClient(NumPyClient):
    """Flower client implementing sample-level DP with Opacus."""

    def __init__(self, train_df, test_df, target_delta, noise_multiplier, max_grad_norm):
        super().__init__()
        self.device = DEVICE
        self.model = Net().to(self.device)
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        # Preprocess training data
        processed_df = preprocess_can_to_binary(train_df)
        X = processed_df.iloc[:, :-1].values
        y = processed_df.iloc[:, -1].values.astype(np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        self.train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
        self.test_df = test_df

    def fit(self, parameters, config):
        """Train the model with differential privacy and return updated weights."""
        print(f"[Client] Using device: {self.device}")
        set_model_weights(self.model, parameters)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        privacy_engine = PrivacyEngine(secure_mode=False)

        # Make model private
        self.model, optimizer, private_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        epsilon = train_with_privacy(
            model=self.model,
            train_loader=private_loader,
            privacy_engine=privacy_engine,
            optimizer=optimizer,
            delta=self.target_delta,
            device=self.device,
        )

        if epsilon:
            print(f"[Client] Epsilon for delta={self.target_delta}: {epsilon:.2f}")
        else:
            print("[Client] Epsilon not available.")

        return get_model_weights(self.model), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        """Evaluate model on test data and return metrics."""
        set_model_weights(self.model, parameters)
        loss, accuracy = evaluate_model(self.model, self.test_df, self.device)
        return loss, len(self.test_df), {"accuracy": accuracy}


def build_client(context: Context) -> NumPyClient:
    """
    Construct and return a configured Flower client with DP settings.

    Args:
        context: Flower context containing node configuration.

    Returns:
        A Flower NumPyClient instance.
    """
    partition_id = context.node_config["partition-id"]

    # Differential privacy configuration
    noise_multiplier = context.run_config["noise-multiplier"]
    max_grad_norm = context.run_config["max-grad-norm"]
    target_delta = context.run_config["target-delta"]

    train_df = load_train_data(f"dataset/train/client{partition_id + 1}")
    test_df = load_test_partition(partition_id, "dataset/test.csv", num_clients=4)

    return DifferentialPrivacyClient(
        train_df=train_df,
        test_df=test_df,
        target_delta=target_delta,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    ).to_client()


# Define the client app
app = ClientApp(client_fn=build_client)
