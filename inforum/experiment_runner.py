"""
LegionITS Experiment Runner — NSL-KDD Edition
===============================================
All experiments run on NSL-KDD (122 features, binary classification).
Prerequisite: run prepare_nslkdd.py first to generate dataset_nslkdd/.

Configurations:
  1. Local baseline   — each client trains independently
  2. FL IID           — FedAvg, balanced label splits
  3. FL Non-IID       — FedAvg, label-skewed [0.1, 0.3, 0.5, 0.7]
  4. FL+DP IID        — FedAvg+DP, ε ∈ {0.5, 1.0, 1.64, 3.0, 5.0}
  5. FL+DP Non-IID    — FedAvg+DP, ε=1.64
  6. MI Attack        — centralized DP demo (300 samples, 500 epochs)
                        showing memorisation → privacy tradeoff

Usage:
  python experiment_runner.py --data_dir dataset_nslkdd --rounds 3 --epochs 10
  python experiment_runner.py --data_dir dataset_nslkdd --rounds 3 --epochs 10 --skip_dp
"""

import argparse
import json
import os
import random
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              confusion_matrix, roc_curve)
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
N_CLIENTS    = 4
BATCH_SIZE   = 512
TARGET_DELTA = 1e-5
INPUT_DIM    = 122          # NSL-KDD features after one-hot encoding
DEFAULT_SEED = 42

# Accountant used for BOTH sigma calibration and epsilon reporting. Opacus'
# PrivacyEngine defaults to "prv"; calibrating with a different mechanism than
# we report with silently mislabels the budget.
ACCOUNTANT   = "prv"
NONIID_RATIOS = [0.1, 0.3, 0.5, 0.7]
EPSILON_SWEEP = [0.5, 1.0, 1.64, 3.0, 5.0]

# NSL-KDD: ~59k normal, ~67k attack in training set
# Non-IID ratios [0.1,0.3,0.5,0.7] at 15k/client needs:
#   normal: 13500+10500+7500+4500 = 36000  ✓ (have 59k)
#   attack:  1500+4500+7500+10500 = 24000  ✓ (have 67k)
NONIID_SAMPLES_PER_CLIENT = 15_000

# MI experiment: small subset to force memorisation
MI_SAMPLES_TOTAL = 300      # 75 per client
MI_BATCH_SIZE    = 16
MI_EPOCHS        = 500


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class Net(nn.Module):
    """MLP for NSL-KDD binary intrusion detection (122 features → 1).
    LayerNorm instead of BatchNorm for Opacus DP compatibility."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.ReLU(), nn.LayerNorm(256), nn.Dropout(0.3),
            nn.Linear(256, 64),        nn.ReLU(), nn.LayerNorm(64),  nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.hidden(x)


def get_weights(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_weights(model, weights):
    """Set model weights, casting each array back to original dtype.
    Handles LayerNorm int64 counters safely."""
    state = model.state_dict()
    sd = OrderedDict({
        k: torch.tensor(v.astype(state[k].cpu().numpy().dtype))
        for k, v in zip(state.keys(), weights)
    })
    model.load_state_dict(sd, strict=True)


def average_weights(weight_list, sizes):
    """Weighted FedAvg. Uses float64 accumulator to handle
    mixed int64/float32 dtypes from LayerNorm buffers."""
    total = sum(sizes)
    avg   = [np.zeros(w.shape, dtype=np.float64) for w in weight_list[0]]
    for weights, n in zip(weight_list, sizes):
        for i, w in enumerate(weights):
            avg[i] += w.astype(np.float64) * (n / total)
    return avg


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_nslkdd(data_dir):
    """Load preprocessed NSL-KDD CSVs. Enforce float32 on all feature cols."""
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    for df in [train, test]:
        for col in df.columns:
            if col != 'label':
                df[col] = (pd.to_numeric(df[col], errors='coerce')
                             .fillna(0).astype(np.float32))
        df['label'] = df['label'].astype(np.int32)
    return train, test


def set_seed(seed=DEFAULT_SEED):
    """Seed every RNG the experiments draw from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader_generator(seed=None):
    """Seeded generator for one DataLoader.

    With seed=None the seed is drawn from torch's global RNG, which set_seed()
    has already fixed: the run stays reproducible while successive loaders
    (and successive repetitions in run_fl_dp_averaged) still shuffle
    differently.
    """
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def df_to_loader(df, shuffle=True, batch_size=BATCH_SIZE, seed=None):
    X = torch.tensor(df.drop(columns=['label']).values.astype(np.float32),
                     dtype=torch.float32)
    y = torch.tensor(df['label'].values.astype(np.float32),
                     dtype=torch.float32).view(-1, 1)
    return DataLoader(TensorDataset(X, y),
                      batch_size=batch_size, shuffle=shuffle,
                      drop_last=True, generator=_loader_generator(seed))


# ─────────────────────────────────────────────
# Data splitting
# ─────────────────────────────────────────────
def iid_split(df, n, n_per_client=None, seed=DEFAULT_SEED):
    """Balanced IID split — equal attack/normal ratio per client."""
    df_n = df[df['label'] == 0].reset_index(drop=True)
    df_a = df[df['label'] == 1].reset_index(drop=True)

    if n_per_client is not None:
        ratio_a = len(df_a) / len(df)
        per_a   = min(int(n_per_client * ratio_a), len(df_a) // n)
        per_n   = min(n_per_client - per_a,        len(df_n) // n)
    else:
        per_n = len(df_n) // n
        per_a = len(df_a) // n

    parts = []
    for i in range(n):
        part = pd.concat([
            df_n.iloc[i * per_n:(i + 1) * per_n],
            df_a.iloc[i * per_a:(i + 1) * per_a],
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
        parts.append(part)
    return parts


def noniid_split(df, ratios, n_per_client, seed=DEFAULT_SEED):
    """Non-IID label-skew split.
    Client i gets fraction ratios[i] of attack (label=1) samples,
    simulating organisations with different threat exposure levels."""
    df_n = df[df['label'] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
    df_a = df[df['label'] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)

    parts = []
    n_off, a_off = 0, 0
    for ratio in ratios:
        n_a = int(n_per_client * ratio)
        n_n = n_per_client - n_a
        assert n_off + n_n <= len(df_n), \
            f"Not enough normal samples (need {n_off+n_n}, have {len(df_n)})"
        assert a_off + n_a <= len(df_a), \
            f"Not enough attack samples (need {a_off+n_a}, have {len(df_a)})"
        part = pd.concat([
            df_n.iloc[n_off:n_off + n_n],
            df_a.iloc[a_off:a_off + n_a],
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
        parts.append(part)
        n_off += n_n
        a_off += n_a
    return parts


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
@dataclass
class Metrics:
    accuracy: float
    f1: float
    recall: float
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0

    def __repr__(self):
        return (f"Acc={self.accuracy:.4f}  F1={self.f1:.4f}  "
                f"Recall={self.recall:.4f}  "
                f"[TN={self.tn} FP={self.fp} FN={self.fn} TP={self.tp}]")


def evaluate(model, test_df):
    model.to(DEVICE).eval()
    X      = torch.tensor(test_df.drop(columns=['label']).values.astype(np.float32)).to(DEVICE)
    y_true = test_df['label'].values.astype(np.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).cpu().numpy().flatten()
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = (probs > thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, int(cm[0, 0]))
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=1)),
        recall=float(recall_score(y_true, y_pred, zero_division=1)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────
def local_train(model, loader, epochs, lr):
    model.to(DEVICE).train()
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(X), y).backward()
            opt.step()
    return model


def dp_train(model, loader, epochs, noise_multiplier, privacy_engine,
             max_grad_norm=1.0):
    """One round of DP-SGD for a single client.

    `privacy_engine` is owned by the caller and reused across rounds, so its
    accountant accumulates the budget instead of restarting every round.
    make_private() only attaches a step hook to the existing accountant — it
    does not clear the history.
    """
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model, opt, private_loader = privacy_engine.make_private(
        module=model, optimizer=opt, data_loader=loader,
        noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for X, y in private_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(X), y).backward()
            opt.step()
    if hasattr(model, '_module'):
        model = model._module
    return model


def get_noise_multiplier_for_epsilon(target_epsilon, sample_rate, epochs=None,
                                     steps=None):
    """Calibrate sigma. Pass exactly one of `epochs` or `steps`."""
    from opacus.accountants.utils import get_noise_multiplier as _gnm
    return _gnm(target_epsilon=target_epsilon, target_delta=TARGET_DELTA,
                sample_rate=sample_rate, epochs=epochs, steps=steps,
                accountant=ACCOUNTANT)


# ─────────────────────────────────────────────
# Experiment 0: Centralised (pooled data) baseline
# ─────────────────────────────────────────────
def run_centralised(train_df, test_df, epochs, lr, label="centralised"):
    """Pooled-data upper bound: one model, all data, no partitioning.

    `epochs` is the TOTAL number of passes over the pooled training set.
    main() passes rounds * epochs so the optimisation budget matches what a
    single FL client receives across every round, making the gap to FL
    attributable to federation rather than to training length.

    Returns the same dict shape as run_fl(), with the single training run
    recorded as one entry in "rounds".
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label} — Centralised (pooled), {epochs} epochs")
    print(f"{'='*60}")
    print(f"  n={len(train_df):,} | attack={train_df['label'].mean():.2f}")

    model  = Net()
    loader = df_to_loader(train_df)
    model  = local_train(model, loader, epochs, lr)
    m      = evaluate(model, test_df)
    print(f"  Centralised model: {m}")

    return {
        "config": label,
        "type":   "centralised",
        "rounds": [{"round": 1, **asdict(m)}],
        "final":  asdict(m),
    }


# ─────────────────────────────────────────────
# Experiment 1: Local baseline
# ─────────────────────────────────────────────
def run_local_baseline(train_parts, test_df, epochs, lr, label="local"):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label} — Local Baseline (no federation)")
    print(f"{'='*60}")
    results = {"config": label, "type": "local", "clients": []}
    all_metrics = []

    for i, part in enumerate(train_parts):
        a_ratio = part['label'].mean()
        print(f"  Client {i+1} | n={len(part):,} | attack={a_ratio:.2f}", end=" → ")
        model  = Net()
        loader = df_to_loader(part)
        model  = local_train(model, loader, epochs, lr)
        m      = evaluate(model, test_df)
        print(m)
        results["clients"].append({"client": i+1, "attack_ratio": round(a_ratio, 2),
                                   **asdict(m)})
        all_metrics.append(m)

    avg = Metrics(
        accuracy=float(np.mean([m.accuracy for m in all_metrics])),
        f1=float(np.mean([m.f1 for m in all_metrics])),
        recall=float(np.mean([m.recall for m in all_metrics])),
        tn=int(sum(m.tn for m in all_metrics)),
        fp=int(sum(m.fp for m in all_metrics)),
        fn=int(sum(m.fn for m in all_metrics)),
        tp=int(sum(m.tp for m in all_metrics)),
    )
    results["average"] = asdict(avg)
    print(f"  → Average: {avg}")
    return results


# ─────────────────────────────────────────────
# Experiment 2: FL (no DP)
# ─────────────────────────────────────────────
def run_fl(train_parts, test_df, rounds, epochs, lr, label="fl"):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label} — FL (no DP), {rounds} rounds, {epochs} epochs/round")
    print(f"{'='*60}")
    results = {"config": label, "type": "fl", "rounds": []}

    global_model = Net()
    loaders      = [df_to_loader(p) for p in train_parts]

    for r in range(1, rounds + 1):
        print(f"\n  Round {r}/{rounds}")
        cw, cs = [], []
        for loader in loaders:
            cm = Net()
            set_weights(cm, get_weights(global_model))
            cm = local_train(cm, loader, epochs, lr)
            cw.append(get_weights(cm))
            cs.append(len(loader.dataset))
        set_weights(global_model, average_weights(cw, cs))
        m = evaluate(global_model, test_df)
        print(f"  Global model: {m}")
        results["rounds"].append({"round": r, **asdict(m)})

    results["final"] = asdict(evaluate(global_model, test_df))
    return results


# ─────────────────────────────────────────────
# Experiment 3: FL+DP (averaged over n runs)
# ─────────────────────────────────────────────
def run_fl_dp(train_parts, test_df, rounds, epochs,
              target_epsilon, label="fl_dp", seed=None):
    """FedAvg + DP-SGD with a privacy budget composed across all rounds.

    One PrivacyEngine per client lives for the whole federation, so each
    client's accountant composes its budget over every round. sigma is
    calibrated against that full budget, at the sample rate Opacus actually
    accounts with (1/len(loader), since the loader drops its last partial
    batch) rather than BATCH_SIZE/min_n.
    """
    from opacus import PrivacyEngine

    print(f"\n  \u2192 FL+DP target \u03b5={target_epsilon} "
          f"(composed over {rounds} rounds)")
    results = {"config": label, "type": "fl_dp",
               "target_epsilon": target_epsilon, "rounds": []}

    global_model = Net()
    # Distinct deterministic loader seed per client, so clients do not share a
    # batch order. seed=None falls back to the global RNG fixed by set_seed().
    loaders = [df_to_loader(p, seed=None if seed is None else seed + i)
               for i, p in enumerate(train_parts)]

    noise_mults, engines = [], []
    for loader in loaders:
        n_batches = len(loader)
        noise_mults.append(get_noise_multiplier_for_epsilon(
            target_epsilon, 1.0 / n_batches,
            steps=rounds * epochs * n_batches))
        engines.append(PrivacyEngine(accountant=ACCOUNTANT, secure_mode=False))

    results["noise_multiplier"] = [round(nm, 4) for nm in noise_mults]
    print(f"     noise_multiplier/client={results['noise_multiplier']}")

    per_round_eps = []
    for r in range(1, rounds + 1):
        print(f"     Round {r}/{rounds}", end=" | ")
        cw, cs, round_eps = [], [], []
        for loader, noise_mult, pe in zip(loaders, noise_mults, engines,
                                          strict=True):
            cm = Net()
            set_weights(cm, get_weights(global_model))
            cm = dp_train(cm, loader, epochs, noise_mult, privacy_engine=pe)
            cw.append(get_weights(cm))
            cs.append(len(loader.dataset))
            round_eps.append(float(pe.get_epsilon(delta=TARGET_DELTA)))
        set_weights(global_model, average_weights(cw, cs))
        m       = evaluate(global_model, test_df)
        avg_eps = float(np.mean(round_eps))
        per_round_eps.append(avg_eps)
        print(f"\u03b5={avg_eps:.4f} cumulative | {m}")
        results["rounds"].append({"round": r, "achieved_epsilon": avg_eps,
                                  **asdict(m)})

    results["final"]             = asdict(evaluate(global_model, test_df))
    results["per_round_epsilon"] = per_round_eps
    results["composed_epsilon"]  = per_round_eps[-1]
    # Total budget after all rounds. This key previously held the mean of the
    # per-round values, which were identical because the accountant reset.
    results["mean_achieved_epsilon"] = per_round_eps[-1]
    return results


def run_fl_dp_averaged(train_parts, test_df, rounds, epochs,
                        target_epsilon, n_runs=3, label="fl_dp", seed=None):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label} — FL+DP (ε={target_epsilon}, {n_runs} runs)")
    print(f"{'='*60}")

    all_finals, all_eps = [], []
    all_nm, all_per_round = [], []
    all_rounds = [[] for _ in range(rounds)]

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}")
        # Offset per repetition so the runs differ in batch order as well as
        # in DP noise, while staying reproducible from --seed.
        result = run_fl_dp(train_parts, test_df, rounds, epochs,
                           target_epsilon, label=f"{label}_run{run}",
                           seed=None if seed is None else seed + 1000 * run)
        all_finals.append(result["final"])
        all_eps.append(result["composed_epsilon"])
        all_nm.append(result["noise_multiplier"])
        all_per_round.append(result["per_round_epsilon"])
        for r_data in result["rounds"]:
            all_rounds[r_data["round"] - 1].append(r_data)

    avg_rounds = []
    for r_idx, rr in enumerate(all_rounds):
        avg_rounds.append({
            "round":            r_idx + 1,
            "accuracy":         float(np.mean([r["accuracy"]         for r in rr])),
            "f1":               float(np.mean([r["f1"]               for r in rr])),
            "recall":           float(np.mean([r["recall"]           for r in rr])),
            "achieved_epsilon": float(np.mean([r["achieved_epsilon"] for r in rr])),
            "accuracy_std":     float(np.std( [r["accuracy"]         for r in rr])),
        })

    averaged = {
        "config":               label,
        "type":                 "fl_dp",
        "target_epsilon":       target_epsilon,
        "n_runs":               n_runs,
        "noise_multiplier":      all_nm[0],
        "composed_epsilon":      float(np.mean(all_eps)),
        "per_round_epsilon":     [float(np.mean(x))
                                 for x in zip(*all_per_round, strict=True)],
        "mean_achieved_epsilon": float(np.mean(all_eps)),
        "std_achieved_epsilon":  float(np.std(all_eps)),
        "rounds": avg_rounds,
        "final": {
            "accuracy":     float(np.mean([f["accuracy"] for f in all_finals])),
            "f1":           float(np.mean([f["f1"]       for f in all_finals])),
            "recall":       float(np.mean([f["recall"]   for f in all_finals])),
            "accuracy_std": float(np.std( [f["accuracy"] for f in all_finals])),
            "f1_std":       float(np.std( [f["f1"]       for f in all_finals])),
            "recall_std":   float(np.std( [f["recall"]   for f in all_finals])),
        },
    }

    f = averaged["final"]
    print(f"\n  Averaged over {n_runs} runs:")
    print(f"    Acc={f['accuracy']:.4f}±{f['accuracy_std']:.4f}  "
          f"F1={f['f1']:.4f}  Recall={f['recall']:.4f}  "
          f"ε={averaged['mean_achieved_epsilon']:.4f}")
    return averaged


# ─────────────────────────────────────────────
# Experiment 4: MI — centralized DP demo
# ─────────────────────────────────────────────
def run_mi_experiment(train_df, test_df):
    """
    Centralized MI demonstration showing the DP memorisation tradeoff.

    Uses MI_SAMPLES_TOTAL (300) training samples and MI_EPOCHS (500)
    to force memorisation without DP, then shows DP prevents it.

    Three settings: no DP, DP ε≈1.0, DP ε≈0.3.
    Primary metric: train-test accuracy gap + loss ratio.
    MI attack metric: balanced threshold attack (Yeom 2018).
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT: Membership Inference — DP Memorisation Demo")
    print(f"  {MI_SAMPLES_TOTAL} training samples | {MI_EPOCHS} epochs | "
          f"batch {MI_BATCH_SIZE}")
    print(f"{'='*60}")

    # Sample balanced training set
    n_each   = MI_SAMPLES_TOTAL // 2
    mi_train = pd.concat([
        train_df[train_df['label']==0].sample(n=n_each, random_state=42),
        train_df[train_df['label']==1].sample(n=n_each, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Large unregularised model for memorisation
    class MINet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, 512), nn.ReLU(),
                nn.Linear(512, 256),       nn.ReLU(),
                nn.Linear(256, 64),        nn.ReLU(),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x)

    def mi_train_plain(model, loader, epochs):
        model.to(DEVICE).train()
        opt     = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        for ep in range(epochs):
            for X, y in loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss_fn(model(X), y).backward()
                opt.step()
            if (ep + 1) % 100 == 0:
                with torch.no_grad():
                    tl = sum(loss_fn(model(X.to(DEVICE)), y.to(DEVICE)).item()
                             for X, y in loader) / len(loader)
                print(f"      epoch {ep+1} | train loss: {tl:.4f}")
        return model

    def mi_train_dp(model, loader, epochs, noise_mult):
        from opacus import PrivacyEngine
        model.to(DEVICE).train()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        pe  = PrivacyEngine(accountant=ACCOUNTANT, secure_mode=False)
        model, opt, pl = pe.make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=noise_mult, max_grad_norm=1.0)
        loss_fn = nn.BCEWithLogitsLoss()
        for ep in range(epochs):
            for X, y in pl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss_fn(model(X), y).backward()
                opt.step()
            if (ep + 1) % 100 == 0:
                m = model._module if hasattr(model, '_module') else model
                m.eval()
                with torch.no_grad():
                    tl = sum(loss_fn(m(X.to(DEVICE)), y.to(DEVICE)).item()
                             for X, y in loader) / len(loader)
                m.train()
                eps = pe.get_epsilon(delta=TARGET_DELTA)
                print(f"      epoch {ep+1} | train loss: {tl:.4f} | ε={eps:.3f}")
        eps = pe.get_epsilon(delta=TARGET_DELTA)
        if hasattr(model, '_module'):
            model = model._module
        return model, float(eps)

    def mi_evaluate(model, df):
        model.to(DEVICE).eval()
        X = torch.tensor(df.drop(columns=['label']).values.astype(np.float32)).to(DEVICE)
        y = df['label'].values.astype(np.float32)
        with torch.no_grad():
            p = torch.sigmoid(model(X)).cpu().numpy().flatten()
        fpr, tpr, ths = roc_curve(y, p)
        t = ths[np.argmax(tpr - fpr)]
        return float(accuracy_score(y, (p > t).astype(int)))

    def threshold_mi_attack(model, member_df, nonmember_df, n=1000):
        """Yeom 2018 threshold attack with balanced-accuracy optimisation."""
        model.to(DEVICE).eval()
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        def losses(df):
            s = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
            X = torch.tensor(s.drop(columns=['label']).values.astype(np.float32)).to(DEVICE)
            y = torch.tensor(s['label'].values.astype(np.float32)).view(-1,1).to(DEVICE)
            with torch.no_grad():
                return loss_fn(model(X), y).cpu().numpy().flatten()

        m_l  = losses(member_df)
        nm_l = losses(nonmember_df)

        # Negate: low loss = high score = likely member
        all_s  = np.concatenate([-m_l, -nm_l])
        all_lb = np.array([1]*len(m_l) + [0]*len(nm_l))
        n_pos, n_neg = (all_lb==1).sum(), (all_lb==0).sum()

        # Maximise balanced accuracy (not simple accuracy — avoids class imbalance trap)
        best_bal, best_t = 0.0, 0.0
        for t in np.linspace(all_s.min(), all_s.max(), 500):
            preds = (all_s > t).astype(int)
            tp_t  = ((preds==1) & (all_lb==1)).sum()
            tn_t  = ((preds==0) & (all_lb==0)).sum()
            bal   = (tp_t/n_pos + tn_t/n_neg) / 2
            if bal > best_bal:
                best_bal, best_t = bal, t

        tp = ((all_s > best_t) & (all_lb==1)).sum()
        tn = ((all_s <= best_t) & (all_lb==0)).sum()
        return {
            "attack_accuracy":     round(float((tp/n_pos + tn/n_neg)/2), 4),
            "member_loss_mean":    round(float(m_l.mean()),  4),
            "nonmember_loss_mean": round(float(nm_l.mean()), 4),
            "loss_ratio":          round(float(nm_l.mean() / max(m_l.mean(), 1e-8)), 1),
        }

    sr = MI_BATCH_SIZE / MI_SAMPLES_TOTAL
    settings = [
        {"label": "no_dp",    "dp": False, "eps": None},
        {"label": "dp_eps1",  "dp": True,  "eps": 1.0},
        {"label": "dp_eps03", "dp": True,  "eps": 0.3},
    ]
    rows, results = [], {}

    for s in settings:
        print(f"\n  ── {s['label'].upper()} ──")
        torch.manual_seed(42)
        model  = MINet()
        loader = DataLoader(
            TensorDataset(
                torch.tensor(mi_train.drop(columns=['label']).values.astype(np.float32)),
                torch.tensor(mi_train['label'].values.astype(np.float32)).view(-1,1)
            ),
            batch_size=MI_BATCH_SIZE, shuffle=True, drop_last=True
        )

        if s['dp']:
            nm = get_noise_multiplier_for_epsilon(s['eps'], sr, MI_EPOCHS)
            print(f"  noise_mult={nm:.2f}")
            model, achieved_eps = mi_train_dp(model, loader, MI_EPOCHS, nm)
        else:
            model = mi_train_plain(model, loader, MI_EPOCHS)
            achieved_eps = None

        tr_acc  = mi_evaluate(model, mi_train)
        te_acc  = mi_evaluate(model, test_df)
        mi_res  = threshold_mi_attack(model, mi_train, test_df)
        eps_str = f"{achieved_eps:.2f}" if achieved_eps else "—"

        print(f"  Train acc: {tr_acc:.4f} | Test acc: {te_acc:.4f} | "
              f"Gap: {tr_acc-te_acc:+.4f}")
        print(f"  Loss ratio: {mi_res['loss_ratio']}× | "
              f"MI accuracy: {mi_res['attack_accuracy']:.4f}")

        rows.append({"label": s['label'], "eps": eps_str,
                     "train_acc": tr_acc, "test_acc": te_acc,
                     "gap": round(tr_acc - te_acc, 4),
                     "mi_accuracy": mi_res['attack_accuracy'],
                     "loss_ratio": mi_res['loss_ratio']})
        results[s['label']] = {**mi_res, "train_acc": tr_acc,
                                "test_acc": te_acc, "achieved_eps": achieved_eps}

    print(f"\n{'─'*66}")
    print(f"{'Setting':<14} {'ε':>6} {'Train':>8} {'Test':>8} "
          f"{'Gap':>7} {'MI Acc':>8} {'Loss ×':>8}")
    print("─"*66)
    for r in rows:
        print(f"{r['label']:<14} {r['eps']:>6} {r['train_acc']:>8.4f} "
              f"{r['test_acc']:>8.4f} {r['gap']:>+7.4f} "
              f"{r['mi_accuracy']:>8.4f} {r['loss_ratio']:>7}×")
    print("─"*66)

    return {"config": "membership_inference", "type": "mi",
            "mi_samples": MI_SAMPLES_TOTAL, "mi_epochs": MI_EPOCHS,
            "results": results, "rows": rows}


# ─────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────
def print_summary_table(all_results):
    print("\n" + "="*80)
    print("SUMMARY TABLE (final / averaged results)")
    print("="*80)
    print(f"{'Config':<35} {'Accuracy':>12} {'F1':>12} "
          f"{'Recall':>12} {'Achieved ε':>12}")
    print("-"*80)
    for r in all_results:
        if r.get("type") == "mi":
            for row in r.get("rows", []):
                label = f"MI {row['label']} (ε={row['eps']})"
                print(f"{label:<35} {row['mi_accuracy']:>12.4f} "
                      f"{'—':>12} {'—':>12} {row['eps']:>12}")
            continue
        final   = r.get("final") or r.get("average", {})
        acc     = final.get("accuracy", 0)
        f1      = final.get("f1", 0)
        rec     = final.get("recall", 0)
        acc_std = final.get("accuracy_std")
        eps     = r.get("mean_achieved_epsilon")
        eps_str = f"{eps:.4f}" if eps is not None else "—"
        acc_str = f"{acc:.4f}±{acc_std:.4f}" if acc_std else f"{acc:.4f}"
        print(f"{r['config']:<35} {acc_str:>12} {f1:>12.4f} "
              f"{rec:>12.4f} {eps_str:>12}")
    print("="*80)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LegionITS Experiment Runner — NSL-KDD")
    p.add_argument("--data_dir", default="dataset_nslkdd")
    p.add_argument("--rounds",   type=int,   default=3)
    p.add_argument("--epochs",   type=int,   default=10)
    p.add_argument("--lr",       type=float, default=0.001)
    p.add_argument("--runs",     type=int,   default=3,
                   help="Repetitions for DP experiments")
    p.add_argument("--subset",   type=int,   default=None,
                   help="Samples/client cap for IID/non-IID (None = full data)")
    p.add_argument("--output",   default="results.json")
    p.add_argument("--seed",     type=int,   default=DEFAULT_SEED,
                   help="Global RNG seed (torch, cuda, numpy, random, splits)")
    p.add_argument("--skip_dp",  action="store_true",
                   help="Skip FL+DP and MI experiments")
    p.add_argument("--mi_only",  action="store_true",
                   help="Run only the MI experiment")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"Device  : {DEVICE}")
    print(f"Dataset : NSL-KDD ({args.data_dir})")
    print(f"Rounds  : {args.rounds} | Epochs/round: {args.epochs} | LR: {args.lr}")
    print(f"Seed    : {args.seed} | Accountant: {ACCOUNTANT}")
    if args.subset:
        print(f"Subset  : {args.subset:,} samples/client")

    # ── Load ──────────────────────────────────
    print("\nLoading NSL-KDD...")
    train_df, test_df = load_nslkdd(args.data_dir)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,} | "
          f"Features: {train_df.shape[1]-1}")
    print(f"  Attack ratio — train: {train_df['label'].mean():.2f}  "
          f"test: {test_df['label'].mean():.2f}")

    # ── Splits ────────────────────────────────
    noniid_n     = args.subset if args.subset else NONIID_SAMPLES_PER_CLIENT
    iid_parts    = iid_split(train_df, N_CLIENTS, n_per_client=args.subset,
                             seed=args.seed)
    noniid_parts = noniid_split(train_df, NONIID_RATIOS, noniid_n,
                                seed=args.seed)

    print("\nIID splits:")
    for i, p in enumerate(iid_parts):
        print(f"  client{i+1}: attack={p['label'].mean():.2f}  n={len(p):,}")
    print("Non-IID splits:")
    for i, p in enumerate(noniid_parts):
        print(f"  client{i+1}: attack={p['label'].mean():.2f}  n={len(p):,}")

    # ── MI-only mode ──────────────────────────
    if args.mi_only:
        print("\n── MI-only mode ──")
        existing = []
        if os.path.exists(args.output):
            with open(args.output) as f:
                existing = json.load(f)
            existing = [r for r in existing if r.get("config") != "membership_inference"]
        existing.append(run_mi_experiment(train_df, test_df))
        print_summary_table(existing)
        with open(args.output, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Results saved to: {args.output}")
        return

    # ── Full run ──────────────────────────────
    all_results = []
    t0 = time.time()

    all_results.append(run_centralised(train_df, test_df,
                                       args.rounds * args.epochs, args.lr))
    all_results.append(run_local_baseline(iid_parts,    test_df, args.epochs, args.lr, "local_iid"))
    all_results.append(run_local_baseline(noniid_parts, test_df, args.epochs, args.lr, "local_noniid"))
    all_results.append(run_fl(iid_parts,    test_df, args.rounds, args.epochs, args.lr, "fl_iid"))
    all_results.append(run_fl(noniid_parts, test_df, args.rounds, args.epochs, args.lr, "fl_noniid"))

    if not args.skip_dp:
        for eps in EPSILON_SWEEP:
            all_results.append(run_fl_dp_averaged(
                iid_parts, test_df, args.rounds, args.epochs,
                target_epsilon=eps, n_runs=args.runs,
                label=f"fl_dp_iid_eps{eps}", seed=args.seed))
        all_results.append(run_fl_dp_averaged(
            noniid_parts, test_df, args.rounds, args.epochs,
            target_epsilon=1.64, n_runs=args.runs,
            label="fl_dp_noniid_eps1.64", seed=args.seed))
        all_results.append(run_mi_experiment(train_df, test_df))

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed/60:.1f} min")
    print_summary_table(all_results)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
