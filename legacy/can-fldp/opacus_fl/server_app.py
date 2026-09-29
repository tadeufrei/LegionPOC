"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import logging
import torch
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from opacus_fl.task import Net, get_model_weights

# Select computation device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set Flower logging level to INFO
logging.getLogger("flwr").setLevel(logging.INFO)

def aggregate_accuracy(metrics_list: list[tuple[int, Metrics]]) -> Metrics:
    """
    Compute weighted average accuracy across clients.

    Args:
        metrics_list: List of (num_examples, metrics_dict) tuples.

    Returns:
        Aggregated metrics dictionary.
    """
    total_samples = 0
    cumulative_accuracy = 0.0

    for num_examples, metrics in metrics_list:
        cumulative_accuracy += num_examples * metrics["accuracy"]
        total_samples += num_examples

    return {"accuracy": cumulative_accuracy / total_samples}

def build_server_components(context: Context) -> ServerAppComponents:
    """
    Build and return Flower ServerAppComponents for a DP-enabled FL server.

    Args:
        context: Flower server context including runtime configuration.

    Returns:
        ServerAppComponents containing strategy and server config.
    """
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model weights
    initial_weights = get_model_weights(Net().to(DEVICE))
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # Define federated averaging strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=aggregate_accuracy,
        initial_parameters=initial_parameters,
    )

    # Define server configuration
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)

# Instantiate the Flower server app
app = ServerApp(server_fn=build_server_components)