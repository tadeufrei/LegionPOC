"""
results_to_tables.py
====================
Reads results.json from experiment_runner.py and prints
plain-text and LaTeX tables for the INForum paper.

Usage:
  python results_to_tables.py --results results.json
"""

import argparse
import json


# ─────────────────────────────────────────────
# Row extraction
# ─────────────────────────────────────────────
def extract_row(result: dict) -> dict:
    """Extract display values from a single experiment result dict."""
    final = result.get("final") or result.get("average", {})

    # Metrics
    acc     = final.get("accuracy", 0.0)
    f1      = final.get("f1",       0.0)
    recall  = final.get("recall",   0.0)
    acc_std = final.get("accuracy_std")
    f1_std  = final.get("f1_std")
    rec_std = final.get("recall_std")

    # Achieved ε — always use the measured value, never the target
    achieved_eps = result.get("mean_achieved_epsilon")
    eps_str = f"{achieved_eps:.2f}" if achieved_eps is not None else "—"

    return {
        "config":   result["config"],
        "label":    _make_label(result),
        "acc":      acc,
        "f1":       f1,
        "recall":   recall,
        "acc_std":  acc_std,
        "f1_std":   f1_std,
        "rec_std":  rec_std,
        "eps":      eps_str,
        "eps_float": achieved_eps,
    }


def _make_label(result: dict) -> str:
    """Build a human-readable label using achieved ε (not target ε)."""
    config = result["config"]
    achieved = result.get("mean_achieved_epsilon")
    eps_str = f"{achieved:.2f}" if achieved is not None else "?"

    static_labels = {
        "local_iid":    "Local (IID)",
        "local_noniid": "Local (Non-IID)",
        "fl_iid":       "FL (IID)",
        "fl_noniid":    "FL (Non-IID)",
    }
    if config in static_labels:
        return static_labels[config]

    if "noniid" in config and "dp" in config:
        return rf"FL+DP (Non-IID, $\varepsilon$={eps_str})"
    if "iid" in config and "dp" in config:
        return rf"FL+DP (IID, $\varepsilon$={eps_str})"

    return config


# ─────────────────────────────────────────────
# Plain-text table
# ─────────────────────────────────────────────
def _fmt(val: float, std: float | None) -> str:
    if std is not None:
        return f"{val:.4f}±{std:.4f}"
    return f"{val:.4f}"


def plain_table(rows: list[dict], title: str) -> None:
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}")
    print(f"  {'Setting':<38} {'Acc':>14} {'F1':>10} {'Recall':>10} {'Achieved ε':>12}")
    print(f"  {'─'*38} {'─'*14} {'─'*10} {'─'*10} {'─'*12}")
    for r in rows:
        # Strip LaTeX from plain label
        label = r["label"].replace(r"$\varepsilon$", "ε").replace(r"\varepsilon", "ε")
        acc_str = _fmt(r["acc"], r.get("acc_std"))
        f1_str  = _fmt(r["f1"],  r.get("f1_std"))
        rec_str = _fmt(r["recall"], r.get("rec_std"))
        print(f"  {label:<38} {acc_str:>14} {f1_str:>10} {rec_str:>10} {r['eps']:>12}")
    print(f"{'─'*80}")


# ─────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────
def _latex_val(val: float, std: float | None) -> str:
    if std is not None:
        return f"${val:.4f} \\pm {std:.4f}$"
    return f"{val:.4f}"


def latex_table(rows: list[dict], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        (r"\textbf{Setting} & \textbf{Accuracy} & \textbf{F1} "
         r"& \textbf{Recall} & \textbf{Achieved $\varepsilon$} \\"),
        r"\hline",
    ]
    for r in rows:
        acc_str = _latex_val(r["acc"], r.get("acc_std"))
        f1_str  = _latex_val(r["f1"],  r.get("f1_std"))
        rec_str = _latex_val(r["recall"], r.get("rec_std"))
        lines.append(
            f"{r['label']} & {acc_str} & {f1_str} & {rec_str} & {r['eps']} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Round-by-round table
# ─────────────────────────────────────────────
def round_table(fl_result: dict, dp_result: dict) -> None:
    print(f"\n{'─'*80}")
    print("  Table 3 — Round-by-round: FL vs FL+DP (IID)")
    print(f"{'─'*80}")
    header = (f"  {'Round':<8} {'FL Acc':>12} {'FL F1':>10} "
              f"{'DP Acc':>14} {'DP F1':>12} {'Achieved ε':>12}")
    print(header)
    print(f"  {'─'*8} {'─'*12} {'─'*10} {'─'*14} {'─'*12} {'─'*12}")

    fl_rounds = {r["round"]: r for r in fl_result.get("rounds", [])}
    for r in dp_result.get("rounds", []):
        rnd = r["round"]
        fl  = fl_rounds.get(rnd, {})
        dp_acc_str = _fmt(r["accuracy"], r.get("accuracy_std"))
        dp_f1_str  = _fmt(r["f1"],       r.get("f1_std"))
        print(f"  {rnd:<8} "
              f"{fl.get('accuracy', 0):>12.4f} "
              f"{fl.get('f1', 0):>10.4f} "
              f"{dp_acc_str:>14} "
              f"{dp_f1_str:>12} "
              f"{r.get('achieved_epsilon', 0):>12.4f}")
    print(f"{'─'*80}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results.json")
    args = p.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    by_config = {r["config"]: r for r in data}

    # ── Table 1: IID — Local vs FL vs FL+DP ε sweep ──
    iid_configs = (
        ["local_iid", "fl_iid"]
        + [f"fl_dp_iid_eps{e}" for e in [0.5, 1.0, 1.64, 3.0, 5.0]]
    )
    iid_rows = [extract_row(by_config[c]) for c in iid_configs if c in by_config]
    plain_table(iid_rows,
                "Table 1 — IID: Local Baseline vs FL vs FL+DP (ε sweep, achieved ε reported)")

    # ── Table 2: IID vs Non-IID ──
    noniid_configs = ["fl_iid", "fl_noniid", "fl_dp_iid_eps1.64", "fl_dp_noniid_eps1.64"]
    noniid_rows = [extract_row(by_config[c]) for c in noniid_configs if c in by_config]
    plain_table(noniid_rows,
                "Table 2 — IID vs Non-IID (FL and FL+DP at operating point)")

    # ── Table 3: Round-by-round ──
    fl_iid = by_config.get("fl_iid")
    dp_iid = by_config.get("fl_dp_iid_eps1.64")
    if fl_iid and dp_iid:
        round_table(fl_iid, dp_iid)

    # ── LaTeX output ──
    print("\n\n" + "=" * 60)
    print("LATEX TABLES")
    print("=" * 60)

    print("\n--- TABLE 1 ---\n")
    print(latex_table(
        iid_rows,
        caption=(
            "Federated learning performance under IID setting. "
            "Local baseline, FL without DP, and FL with DP across privacy budgets. "
            r"Accuracy and F1 reported as mean$\pm$std over "
            + str(next((r.get("n_runs", 3) for r in data if r.get("type") == "fl_dp"), 3))
            + " independent runs for DP configurations."
        ),
        label="tab:iid_results",
    ))

    print("\n--- TABLE 2 ---\n")
    print(latex_table(
        noniid_rows,
        caption=(
            "Impact of data heterogeneity on federated learning performance. "
            "IID and Non-IID label-skewed settings compared for FL and FL+DP "
            r"at the operating point ($\varepsilon \approx 1.25$)."
        ),
        label="tab:noniid_results",
    ))

    # ── Membership Inference ──
    mi = by_config.get("membership_inference")
    if mi:
        r = mi["results"]
        fl  = r["fl_no_dp"]
        dp  = r["fl_dp_eps1_25"]
        print(f"\n{'─'*80}")
        print("  Table 4 — Membership Inference Attack Accuracy")
        print(f"{'─'*80}")
        print(f"  {'Model':<30} {'Attack Acc':>12} {'Member Loss':>14} {'Non-member Loss':>18}")
        print(f"  {'─'*30} {'─'*12} {'─'*14} {'─'*18}")
        print(f"  {'FL (no DP)':<30} "
              f"{fl['attack_accuracy']:>12.4f} "
              f"{fl['member_loss_mean']:>7.4f}±{fl['member_loss_std']:.4f} "
              f"{fl['nonmember_loss_mean']:>10.4f}±{fl['nonmember_loss_std']:.4f}")
        print(f"  {'FL+DP (ε≈1.25)':<30} "
              f"{dp['attack_accuracy']:>12.4f} "
              f"{dp['member_loss_mean']:>7.4f}±{dp['member_loss_std']:.4f} "
              f"{dp['nonmember_loss_mean']:>10.4f}±{dp['nonmember_loss_std']:.4f}")
        print(f"  Random baseline (ideal DP): 0.5000")
        print(f"{'─'*80}")


if __name__ == "__main__":
    main()
