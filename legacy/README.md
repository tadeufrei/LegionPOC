# Legacy: CAN-bus experiments

Flower + PyTorch federated learning on the Han et al. CAN-bus intrusion
dataset, with and without Opacus differential privacy. This is the artifact
for the arXiv preprint (2512.14242), superseded by the NSL-KDD experiments
in `inforum/`.

- `can-fl/`   — FedAvg, no DP
- `can-fldp/` — FedAvg + Opacus DP-SGD (fixed noise_multiplier=0.8)

Model: 77 -> 256 -> 64 -> 1, ReLU + LayerNorm + Dropout(0.3).
4 clients, 3 rounds, 10 local epochs, batch 512, lr 1e-3.

Retained as a cross-domain generalisation result. The datasets are committed
here as they came with the original snapshot.
