# TFG Prototype

This repository contains a binary-classification prototype based on a
Michigan-style Evolution Strategy (ES) for the hidden layer of an MLP.
Each ES individual represents one hidden neuron. The output layer is solved
as an unregularised logistic classifier. The final score is a logit, so the
model returns p(y=+1|X) and p(y=-1|X)=1-p(y=+1|X).

## Main Flow

1. `data.py` loads a numeric binary dataset from OpenML, performs a stratified
   train/test split, imputes missing values on the training split only, and
   standardises features without leaking test statistics.
2. `core_es.py` initialises and mutates the Michigan-style population.
3. `selection.py` selects the hidden neurons that survive each generation.
   The default strategy is OMP-style residual selection.
4. `logistic_output.py` fits the output layer with logistic loss using
   Newton/IRLS, without a regularisation parameter or GCV.
5. `train.py` orchestrates the full loop and can save model arrays plus a JSON
   trace of metrics.

When a validation split is enabled, `train.py` now keeps a checkpoint for the
generation with the best validation accuracy, using validation log-loss as a
tie-breaker. Final test metrics are computed from that selected checkpoint,
not blindly from the last generation. Use `--no-select-best-validation` to keep
the final generation instead.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main ES prototype:

```bash
python train.py --strategy omp --data-id 59 --epochs 100 --save-dir runs
```

Run an ablation:

```bash
python train.py --strategy r2 --data-id 59 --epochs 100
python train.py --strategy cfs --data-id 59 --epochs 100
python train.py --strategy cfs_auc --data-id 59 --epochs 100
```

`cfs_auc` keeps the CFS subset-merit structure, but replaces Pearson
neuron-target relevance with squared Gini-AUC and replaces Pearson
neuron-neuron redundancy with squared Spearman correlation.

Run the memorisation/overfit demo:

```bash
python train.py --overfit --overfit-n-samples 25 --overfit-mu 50 --epochs 150
```

Run the Level-1 benchmark:

```bash
python benchmark_level1.py
```

Compare the four ES selection strategies on the same train/validation/test
split and save plots:

```bash
python compare_strategies.py --epochs 100 --output-dir runs/strategy_comparison
```

Run a multi-seed/multi-dataset comparison:

```bash
python compare_strategies.py \
  --datasets ionosphere sonar breast-w \
  --seeds 42 43 44 \
  --epochs 100 \
  --output-dir runs/strategy_comparison_multi
```

The comparison script writes:

- `strategy_comparison.png`: training-accuracy curves plus final
  validation/test accuracy bars.
- `validation_accuracy.png`: validation-accuracy curves with the selected
  checkpoint marked.
- `strategy_diagnostics.png`: train-validation gap, train log-loss, Newton
  step norm, and mean selection cost.
- `aggregate_accuracy.png`: validation/test accuracy means and standard
  deviations across seeds.
- `aggregate_probabilistic_metrics.png`: aggregate test log-loss, Brier score,
  and ROC-AUC.
- `strategy_comparison_all_runs.json` and `strategy_comparison_summary.json`:
  raw multi-run metrics and aggregate summaries.

Use validation accuracy for strategy selection, and keep test accuracy as the
final held-out estimate.

## Dataset Selection

By default, `data.py` uses the numeric OpenML `data_id`. Passing
`--data-name` overrides the ID and pins the named dataset version:

```bash
python train.py --data-name ionosphere --data-version 1
```

This avoids benchmark mistakes where multiple OpenML IDs accidentally load the
same named dataset.

OpenML cache files are written to `.openml_cache/` by default, keeping external
dataset artifacts inside the project workspace. Override this with
`--data-home` if needed.

## Outputs

When `--save-dir` is provided, `train.py` writes:

- `.npz`: final `alpha`, `L`, and `beta` arrays.
- `.json`: run configuration, final metrics, and per-epoch history.

These files make results traceable without changing the training API.
