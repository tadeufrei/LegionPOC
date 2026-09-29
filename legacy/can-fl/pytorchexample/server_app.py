"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from pytorchexample.task import Net, get_weights

# Set the appropriate device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def aggregate_accuracy(metrics_list: list[tuple[int, Metrics]]) -> Metrics:
    """
    Compute weighted average accuracy from clients.

    Args:
        metrics_list: List of tuples (num_examples, metrics_dict) from each client.

    Returns:
        Dictionary with aggregated accuracy.
    """
    total_samples = 0
    cumulative_accuracy = 0.0

    for num_examples, metrics in metrics_list:
        cumulative_accuracy += num_examples * metrics["accuracy"]
        total_samples += num_examples

    return {"accuracy": cumulative_accuracy / total_samples}


def build_server_components(context: Context) -> ServerAppComponents:
    """
    Define server components including strategy and configuration.

    Args:
        context: Server context with run-time configuration.

    Returns:
        Configured ServerAppComponents object.
    """
    # Retrieve number of federated training rounds from the context
    num_training_rounds = context.run_config["num-server-rounds"]

    # Initialize model and extract weights
    model = Net().to(DEVICE)
    initial_weights = get_weights(model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # Set up the federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=aggregate_accuracy,
        initial_parameters=initial_parameters,
    )

    # Set the server configuration
    server_config = ServerConfig(num_rounds=num_training_rounds)

    return ServerAppComponents(strategy=strategy, config=server_config)


# Create the Flower server app
app = ServerApp(server_fn=build_server_components)
