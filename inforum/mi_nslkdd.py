"""
mi_nslkdd.py
============
Membership inference privacy-utility tradeoff experiment on NSL-KDD.

Runs two MI attacks at three privacy settings to demonstrate the
privacy-utility tradeoff that justifies DP in LegionITS:

  Setting 1: FL without DP          — most vulnerable, best utility
  Setting 2: FL+DP (ε≈1.61)        — partial privacy, good utility
  Setting 3: FL+DP (ε≈0.30)        — strong privacy, reduced utility

Attacks:
  1. Black-box RF    — Random Forest on prediction features
  2. White-box norm  — Per-sample gradient norm at final model

The tradeoff curve shows that DP CAN fully suppress MI (at ε≈0.30),
confirming the mechanism works. ε≈1.61 is LegionITS's operating
point, accepting partial empirical MI risk for operational utility.

Usage:
  python mi_nslkdd.py --data_dir dataset_nslkdd --output results.json
"""

import argparse
import json
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE         = 64
TARGET_DELTA       = 1e-5
N_CLIENTS          = 4
SAMPLES_PER_CLIENT = 500
MI_EPOCHS          = 100
FL_ROUNDS          = 3

# Three privacy settings for the tradeoff table
PRIVACY_SETTINGS = [
    {"label": "FL (no DP)",       "dp": False, "target_eps": None},
    {"label": "FL+DP (ε≈1.61)",   "dp": True,  "target_eps": 1.64},
    {"label": "FL+DP (ε≈0.30)",   "dp": True,  "target_eps": 0.30},
]


# ── Model ──────────────────────────────────────────────────────────────
class Net(nn.Module):
    """MLP for NSL-KDD. LayerNorm (Opacus-compatible). No Dropout."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64),        nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)


def get_weights(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]

def set_weights(model, weights):
    state = model.state_dict()
    sd = OrderedDict({
        k: torch.tensor(v.astype(state[k].cpu().numpy().dtype))
        for k, v in zip(state.keys(), weights)
    })
    model.load_state_dict(sd, strict=True)

def average_weights(wlist, sizes):
    total = sum(sizes)
    avg   = [np.zeros(w.shape, dtype=np.float64) for w in wlist[0]]
    for ws, n in zip(wlist, sizes):
        for i, w in enumerate(ws):
            avg[i] += w.astype(np.float64) * (n / total)
    return avg


# ── Data ───────────────────────────────────────────────────────────────
def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    for df in [train, test]:
        for col in df.columns:
            if col != 'label':
                df[col] = (pd.to_numeric(df[col], errors='coerce')
                             .fillna(0).astype(np.float32))
        df['label'] = df['label'].astype(np.int32)
    return train, test

def split_clients(df, n, n_per_client):
    df_n = df[df['label']==0].sample(frac=1,random_state=42).reset_index(drop=True)
    df_a = df[df['label']==1].sample(frac=1,random_state=42).reset_index(drop=True)
    ratio_a = len(df_a) / len(df)
    per_a   = min(int(n_per_client * ratio_a), len(df_a) // n)
    per_n   = min(n_per_client - per_a,        len(df_n) // n)
    parts   = []
    for i in range(n):
        part = pd.concat([
            df_n.iloc[i*per_n:(i+1)*per_n],
            df_a.iloc[i*per_a:(i+1)*per_a],
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        parts.append(part)
    return parts

def df_to_loader(df):
    X = torch.tensor(df.drop(columns=['label']).values.astype(np.float32),
                     dtype=torch.float32)
    y = torch.tensor(df['label'].values.astype(np.float32),
                     dtype=torch.float32).view(-1, 1)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE,
                      shuffle=True, drop_last=True)

def evaluate(model, df):
    model.to(DEVICE).eval()
    X      = torch.tensor(df.drop(columns=['label']).values.astype(np.float32)).to(DEVICE)
    y_true = df['label'].values.astype(np.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).cpu().numpy().flatten()
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = (probs > thresh).astype(int)
    return (float(accuracy_score(y_true, y_pred)),
            float(f1_score(y_true, y_pred, zero_division=1)),
            float(recall_score(y_true, y_pred, zero_division=1)))


# ── Training ───────────────────────────────────────────────────────────
def local_train(model, loader, epochs, lr=0.001):
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

def dp_train(model, loader, epochs, noise_multiplier, max_grad_norm=1.0):
    from opacus import PrivacyEngine
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    pe  = PrivacyEngine(secure_mode=False)
    model, opt, private_loader = pe.make_private(
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
    eps = pe.get_epsilon(delta=TARGET_DELTA)
    if hasattr(model, '_module'):
        model = model._module
    return model, float(eps)

def get_noise_multiplier(target_eps, sample_rate, epochs):
    from opacus.accountants.utils import get_noise_multiplier as _gnm
    return _gnm(target_epsilon=target_eps, target_delta=TARGET_DELTA,
                sample_rate=sample_rate, epochs=epochs, accountant="rdp")


# ── FL training (one full experiment) ─────────────────────────────────
def train_fl(loaders, input_dim, test_df, use_dp, noise_mult=None):
    """Train FL model for FL_ROUNDS rounds. Returns (model, achieved_eps)."""
    global_model = Net(input_dim)
    achieved_eps  = None
    for r in range(1, FL_ROUNDS + 1):
        cw, cs, eps_list = [], [], []
        for loader in loaders:
            cm = Net(input_dim)
            set_weights(cm, get_weights(global_model))
            if use_dp:
                cm, eps = dp_train(cm, loader, MI_EPOCHS, noise_mult)
                eps_list.append(eps)
            else:
                cm = local_train(cm, loader, MI_EPOCHS)
            cw.append(get_weights(cm))
            cs.append(len(loader.dataset))
        set_weights(global_model, average_weights(cw, cs))
        acc, f1, rec = evaluate(global_model, test_df)
        eps_str = f"ε={np.mean(eps_list):.4f} | " if eps_list else ""
        print(f"    Round {r}/{FL_ROUNDS} | {eps_str}"
              f"Acc={acc:.4f}  F1={f1:.4f}  Recall={rec:.4f}")
        if eps_list:
            achieved_eps = float(np.mean(eps_list))
    return global_model, achieved_eps


# ── Attack 1: Black-box RF ─────────────────────────────────────────────
def rf_mi_attack(model, train_df, test_df, n_samples=600):
    """Black-box Random Forest MI attack on prediction features."""
    model.to(DEVICE).eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def get_features(df):
        s   = df.sample(n=min(n_samples,len(df)),random_state=42).reset_index(drop=True)
        X   = torch.tensor(s.drop(columns=['label']).values.astype(np.float32)).to(DEVICE)
        y   = torch.tensor(s['label'].values.astype(np.float32)).view(-1,1).to(DEVICE)
        with torch.no_grad():
            logits = model(X)
            probs  = torch.sigmoid(logits)
            losses = loss_fn(logits, y)
        p    = probs.cpu().numpy().flatten()
        l    = losses.cpu().numpy().flatten()
        lab  = y.cpu().numpy().flatten()
        conf = np.abs(p - 0.5)
        ent  = -(p*np.log(p+1e-8) + (1-p)*np.log(1-p+1e-8))
        return np.column_stack([l, p, conf, ent, lab])

    mf   = get_features(train_df)
    nmf  = get_features(test_df)
    X_a  = np.vstack([mf, nmf])
    y_a  = np.array([1]*len(mf) + [0]*len(nmf))
    clf  = RandomForestClassifier(n_estimators=100, max_depth=5,
                                   class_weight='balanced', random_state=42)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sc   = cross_val_score(clf, X_a, y_a, cv=cv, scoring='balanced_accuracy')
    m_l  = mf[:,0];  n_l = nmf[:,0]
    return {
        "attack_accuracy":     round(float(sc.mean()), 4),
        "attack_accuracy_std": round(float(sc.std()),  4),
        "member_loss_mean":    round(float(m_l.mean()), 4),
        "nonmember_loss_mean": round(float(n_l.mean()), 4),
        "loss_ratio":          round(float(n_l.mean() / (m_l.mean()+1e-8)), 1),
    }


# ── Attack 2: White-box gradient norm ─────────────────────────────────
def gradient_norm_mi_attack(model, train_df, test_df, n_samples=400):
    """
    White-box gradient norm MI attack.
    Low ||∇L(x,θ)|| = memorised = member.
    DP prevents memorisation → norms increase → distributions overlap.
    """
    input_dim = train_df.shape[1] - 1

    def get_norms(df):
        s = df.sample(n=min(n_samples,len(df)),random_state=42).reset_index(drop=True)
        X = torch.tensor(s.drop(columns=['label']).values.astype(np.float32),
                         dtype=torch.float32)
        y = torch.tensor(s['label'].values.astype(np.float32),
                         dtype=torch.float32).view(-1,1)
        norms = []
        for i in range(len(X)):
            m = Net(input_dim)
            set_weights(m, get_weights(model))
            m.to(DEVICE).train()
            m.zero_grad()
            nn.BCEWithLogitsLoss()(m(X[i:i+1].to(DEVICE)),
                                   y[i:i+1].to(DEVICE)).backward()
            norm = sum(p.grad.detach().norm().item()**2
                       for p in m.parameters()
                       if p.grad is not None)**0.5
            norms.append(norm)
        return np.array(norms)

    member_norms    = get_norms(train_df)
    nonmember_norms = get_norms(test_df)

    all_scores = np.concatenate([-member_norms, -nonmember_norms])
    all_labels = np.array([1]*len(member_norms) + [0]*len(nonmember_norms))

    thresholds = np.linspace(all_scores.min(), all_scores.max(), 300)
    best_acc, best_thresh = 0.0, 0.0
    for t in thresholds:
        preds = (all_scores > t).astype(int)
        acc   = (preds == all_labels).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, t

    tp  = ((all_scores > best_thresh) & (all_labels == 1)).sum()
    tn  = ((all_scores <= best_thresh) & (all_labels == 0)).sum()
    tpr = tp / (all_labels == 1).sum()
    tnr = tn / (all_labels == 0).sum()

    return {
        "attack_accuracy":          round(float((tpr+tnr)/2), 4),
        "member_grad_norm_mean":    round(float(member_norms.mean()), 4),
        "nonmember_grad_norm_mean": round(float(nonmember_norms.mean()), 4),
        "norm_ratio":               round(float(nonmember_norms.mean() /
                                               max(member_norms.mean(), 1e-8)), 1),
    }


# ── Main ───────────────────────────────────────────────────────────────
def run(args):
    print(f"Device : {DEVICE}")
    print(f"Config : {N_CLIENTS} clients × {SAMPLES_PER_CLIENT} samples | "
          f"{MI_EPOCHS} epochs/round | {FL_ROUNDS} rounds")

    print("\nLoading NSL-KDD...")
    train_df, test_df = load_data(args.data_dir)
    input_dim         = train_df.shape[1] - 1
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,} | "
          f"Features: {input_dim}")

    client_parts = split_clients(train_df, N_CLIENTS, SAMPLES_PER_CLIENT)
    pooled_train = pd.concat(client_parts, ignore_index=True)
    loaders      = [df_to_loader(p) for p in client_parts]
    print(f"  Training pool: {len(pooled_train):,} samples")

    # Pre-compute noise multipliers
    sr = BATCH_SIZE / min(len(p) for p in client_parts)
    noise_mults = {}
    for s in PRIVACY_SETTINGS:
        if s["dp"]:
            nm = get_noise_multiplier(s["target_eps"], sr, MI_EPOCHS)
            noise_mults[s["target_eps"]] = nm
            print(f"  noise_mult (ε={s['target_eps']}) = {nm:.4f}")

    all_rows = []   # for tradeoff table

    for setting in PRIVACY_SETTINGS:
        label    = setting["label"]
        use_dp   = setting["dp"]
        t_eps    = setting["target_eps"]
        nm       = noise_mults.get(t_eps)

        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")

        model, achieved_eps = train_fl(loaders, input_dim, test_df,
                                        use_dp, nm)
        final_acc, final_f1, _ = evaluate(model, test_df)

        print(f"\n  Running black-box RF attack...")
        rf = rf_mi_attack(model, pooled_train, test_df)
        print(f"  RF accuracy : {rf['attack_accuracy']:.4f} "
              f"± {rf['attack_accuracy_std']:.4f}  "
              f"loss ratio: {rf['loss_ratio']}×")

        print(f"  Running white-box gradient norm attack...")
        gn = gradient_norm_mi_attack(model, pooled_train, test_df)
        print(f"  GN accuracy : {gn['attack_accuracy']:.4f}  "
              f"norm ratio: {gn['norm_ratio']}×")

        all_rows.append({
            "label":         label,
            "achieved_eps":  achieved_eps,
            "model_acc":     round(final_acc, 4),
            "model_f1":      round(final_f1,  4),
            "rf":            rf,
            "gradient_norm": gn,
        })

    # ── Privacy-utility tradeoff table ────────────────────────────────
    print(f"\n{'='*72}")
    print("PRIVACY-UTILITY TRADEOFF TABLE")
    print(f"{'='*72}")
    h = (f"{'Setting':<22} {'ε':>6} {'Model Acc':>10} "
         f"{'RF Atk':>8} {'GN Atk':>8} "
         f"{'Loss ratio':>11} {'Random':>8}")
    print(h)
    print("-"*72)
    for r in all_rows:
        eps_str = f"{r['achieved_eps']:.2f}" if r['achieved_eps'] else "—"
        print(f"{r['label']:<22} {eps_str:>6} {r['model_acc']:>10.4f} "
              f"{r['rf']['attack_accuracy']:>8.4f} "
              f"{r['gradient_norm']['attack_accuracy']:>8.4f} "
              f"{r['rf']['loss_ratio']:>10}× "
              f"{'0.5000':>8}")
    print("="*72)

    # ── Save ──────────────────────────────────────────────────────────
    mi_result = {
        "config":  "membership_inference",
        "type":    "mi",
        "dataset": "NSL-KDD",
        "settings": all_rows,
    }

    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        existing = [r for r in existing
                    if r.get("config") != "membership_inference"]
    else:
        existing = []

    existing.append(mi_result)
    with open(args.output, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="dataset_nslkdd")
    p.add_argument("--output",   default="results.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
