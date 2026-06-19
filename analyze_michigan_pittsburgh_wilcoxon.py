"""
Wilcoxon paired signed-rank analysis for the Michigan--Pittsburgh experiment.

The script uses individual test metrics, not aggregate means. Pairs are built
by matching dataset label and random seed. Positive deltas are defined as
Michigan minus Pittsburgh, so positive values favour Michigan.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.stats import rankdata, wilcoxon


MATCHED_PATH = Path("runs/michigan_pittsburgh/michigan_pittsburgh_summary.json")
FULL_PATH = Path("runs/full_network_pittsburgh/full_network_pittsburgh_summary.json")
OUTPUT_DIR = Path("runs/michigan_pittsburgh")
ALPHA = 0.05


COMPARISONS = [
    {
        "key": "full_pittsburgh",
        "label": "Michigan vs. Pittsburgh ES",
        "source": "full",
        "pittsburgh_arch": "full_pittsburgh_large",
    },
    {
        "key": "matched_budget_pittsburgh",
        "label": "Michigan vs. Pittsburgh matched-budget",
        "source": "matched",
        "pittsburgh_arch": "pittsburgh_matched",
    },
]

METRICS = [
    ("test_acc", "Test accuracy", "primary"),
    ("test_roc_auc", "Test ROC-AUC", "secondary"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["dataset"]["label"]), int(row["seed"])


def metric_index(rows: list[dict[str, Any]], architecture: str) -> dict[tuple[str, int], dict[str, float]]:
    indexed: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        if row["architecture"] != architecture:
            continue
        metrics = row["metrics"]
        indexed[run_key(row)] = {
            "test_acc": float(metrics["test_acc"]),
            "test_roc_auc": float(metrics["test_roc_auc"]),
        }
    return indexed


def rank_biserial_effect(deltas: np.ndarray) -> float:
    nonzero = deltas[deltas != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    total = float(np.sum(ranks))
    return (positive - negative) / total if total > 0 else 0.0


def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def fmt(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"${value:.3f}$"


def interpretation(metric: str, p_value: float, effect: float) -> str:
    direction = "Michigan" if effect > 0 else "Pittsburgh" if effect < 0 else "neither model"
    if p_value < ALPHA and direction == "Michigan":
        return "Significant, favours Michigan"
    if p_value < ALPHA and direction == "Pittsburgh":
        return "Significant, favours Pittsburgh"
    return f"Not significant, favours {direction}"


def build_pairs(
    michigan: dict[tuple[str, int], dict[str, float]],
    pittsburgh: dict[tuple[str, int], dict[str, float]],
    comparison: dict[str, str],
    metric: str,
) -> tuple[list[dict[str, Any]], list[tuple[str, int]], list[tuple[str, int]]]:
    michigan_keys = set(michigan)
    pittsburgh_keys = set(pittsburgh)
    valid_keys = sorted(michigan_keys & pittsburgh_keys)
    missing_pittsburgh = sorted(michigan_keys - pittsburgh_keys)
    missing_michigan = sorted(pittsburgh_keys - michigan_keys)

    pairs = []
    for dataset, seed in valid_keys:
        m_value = michigan[(dataset, seed)][metric]
        p_value = pittsburgh[(dataset, seed)][metric]
        pairs.append(
            {
                "comparison": comparison["label"],
                "comparison_key": comparison["key"],
                "metric": metric,
                "dataset": dataset,
                "seed": seed,
                "michigan": m_value,
                "pittsburgh": p_value,
                "delta": m_value - p_value,
            }
        )
    return pairs, missing_michigan, missing_pittsburgh


def analyse() -> dict[str, Any]:
    matched_payload = load_json(MATCHED_PATH)
    full_payload = load_json(FULL_PATH)

    michigan = metric_index(matched_payload["runs"], "michigan_cfs")
    matched_pittsburgh = metric_index(matched_payload["runs"], "pittsburgh_matched")
    full_pittsburgh = metric_index(full_payload["runs"], "full_pittsburgh_large")

    pittsburgh_sources = {
        "matched": matched_pittsburgh,
        "full": full_pittsburgh,
    }

    all_pairs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for comparison in COMPARISONS:
        pittsburgh = pittsburgh_sources[comparison["source"]]
        for metric, metric_label, metric_role in METRICS:
            pairs, missing_michigan, missing_pittsburgh = build_pairs(
                michigan,
                pittsburgh,
                comparison,
                metric,
            )
            if missing_michigan or missing_pittsburgh:
                print(f"Missing pairs for {comparison['label']} / {metric}:")
                print(f"  Missing Michigan: {missing_michigan}")
                print(f"  Missing Pittsburgh: {missing_pittsburgh}")
            deltas = np.array([pair["delta"] for pair in pairs], dtype=np.float64)
            if deltas.size == 0:
                raise RuntimeError(f"No valid pairs for {comparison['label']} / {metric}.")
            zero_count = int(np.sum(deltas == 0))
            method = "auto"
            test = wilcoxon(
                deltas,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method=method,
            )
            result = {
                "comparison": comparison["label"],
                "comparison_key": comparison["key"],
                "metric": metric,
                "metric_label": metric_label,
                "metric_role": metric_role,
                "n_pairs": int(deltas.size),
                "zero_differences": zero_count,
                "median_delta": float(median(deltas)),
                "wilcoxon_w": float(test.statistic),
                "p_value": float(test.pvalue),
                "rank_biserial_effect": float(rank_biserial_effect(deltas)),
                "direction": "Michigan" if np.median(deltas) > 0 else "Pittsburgh" if np.median(deltas) < 0 else "Tie",
                "zero_method": "wilcox",
                "alternative": "two-sided",
                "method": method,
            }
            results.append(result)
            all_pairs.extend(pairs)

    secondary_indices = [
        idx for idx, result in enumerate(results) if result["metric_role"] == "secondary"
    ]
    adjusted_secondary = holm_bonferroni([results[idx]["p_value"] for idx in secondary_indices])
    for idx, adjusted in zip(secondary_indices, adjusted_secondary):
        results[idx]["adjusted_p_value"] = float(adjusted)
    for result in results:
        if "adjusted_p_value" not in result:
            result["adjusted_p_value"] = None

    return {
        "alpha": ALPHA,
        "pairing": "dataset and seed",
        "delta_definition": "Michigan metric minus Pittsburgh metric",
        "pairs": all_pairs,
        "results": results,
    }


def write_pairs_csv(pairs: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "comparison",
        "comparison_key",
        "metric",
        "dataset",
        "seed",
        "michigan",
        "pittsburgh",
        "delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)


def write_latex_table(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Wilcoxon paired signed-rank tests for the Michigan--Pittsburgh comparison. Observations are paired by dataset and seed, $\\Delta$ is computed as Michigan minus Pittsburgh, all metrics are test metrics, and the test is bilateral. Adjusted p-values are Holm--Bonferroni corrected only for the secondary ROC-AUC analyses.}",
        "\\label{tab:wilcoxon_michigan_pittsburgh}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrrrl}",
        "\\toprule",
        "Comparison & Metric & $n$ & Median $\\Delta$ & $W$ & p-value & Adjusted p-value & Rank-biserial effect & Interpretation \\\\",
        "\\midrule",
    ]
    for result in results:
        adjusted = "--" if result["adjusted_p_value"] is None else fmt(result["adjusted_p_value"])
        lines.append(
            f"{result['comparison']} & "
            f"{result['metric_label']} & "
            f"{result['n_pairs']} & "
            f"${result['median_delta']:.3f}$ & "
            f"${result['wilcoxon_w']:.3f}$ & "
            f"{fmt(result['p_value'])} & "
            f"{adjusted} & "
            f"${result['rank_biserial_effect']:.3f}$ & "
            f"{interpretation(result['metric'], result['p_value'], result['rank_biserial_effect'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n## Wilcoxon paired signed-rank analysis")
    print("Delta = Michigan metric - Pittsburgh metric; positive favours Michigan.")
    for result in results:
        adjusted = "NA" if result["adjusted_p_value"] is None else f"{result['adjusted_p_value']:.6f}"
        print(
            f"{result['comparison']} | {result['metric_label']} | "
            f"n={result['n_pairs']} | median_delta={result['median_delta']:.6f} | "
            f"W={result['wilcoxon_w']:.6f} | p={result['p_value']:.6f} | "
            f"p_adj={adjusted} | r_rb={result['rank_biserial_effect']:.6f} | "
            f"zeros={result['zero_differences']}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = analyse()
    write_pairs_csv(payload["pairs"], OUTPUT_DIR / "wilcoxon_pairs.csv")
    (OUTPUT_DIR / "wilcoxon_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_latex_table(payload["results"], OUTPUT_DIR / "wilcoxon_table.tex")
    print_summary(payload["results"])
    print(f"\nSaved pairs: {OUTPUT_DIR / 'wilcoxon_pairs.csv'}")
    print(f"Saved results: {OUTPUT_DIR / 'wilcoxon_results.json'}")
    print(f"Saved LaTeX table: {OUTPUT_DIR / 'wilcoxon_table.tex'}")


if __name__ == "__main__":
    main()
