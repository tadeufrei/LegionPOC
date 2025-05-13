# LegionPOC

**LegionPOC** is a proof-of-concept system built using the [Flower](https://flower.dev) federated learning framework. It focuses on secure, privacy-preserving collaborative learning across distributed clients, particularly in the context of **cybersecurity for critical infrastructure**.

LegionPOC integrates:
- **Federated Learning (FL)** using Flower and PyTorch
- **Sample-level Differential Privacy (DP)** with Opacus
- **Binary-encoded CAN bus datasets** for intrusion detection
- **Client-specific data preprocessing** and test partitioning
- **Custom evaluation metrics and threat modeling**

---

## 📦 Key Features

- ✅ Flower-based FL client/server architecture
- ✅ Opacus privacy engine with adjustable ε/δ and gradient clipping
- ✅ Binary conversion of CAN ID, payload, and metadata
- ✅ Fine-grained, label-balanced data partitioning across clients
- ✅ Client-local training and evaluation using accuracy, F1, recall

---

## 🧱 System Architecture

```text
+----------------+      +------------------------+      +----------------+
|  Client 1      | <--->|                        |<---> |  Client N      |
|  - CAN data    |      |     Flower Server      |      |  - CAN data    |
|  - Opacus DP   |      |   (FedAvg Strategy)    |      |  - Opacus DP   |
+----------------+      |                        |      +----------------+
                        +------------------------+
