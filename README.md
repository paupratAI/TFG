# TFG Prototype

This repository contains a binary-classification prototype based on a
Michigan-style Evolution Strategy (ES) for the hidden layer of an MLP.
Each ES individual represents one hidden neuron. The output layer is solved
as an L2-regularised logistic classifier.  The final score is a logit, so the
model returns p(y=+1|X) and p(y=-1|X)=1-p(y=+1|X).

## Main Flow

1. `data.py` loads a numeric binary dataset from OpenML, performs a stratified
   train/test split, imputes missing values on the training split only, and
   standardises features without leaking test statistics.
2. `core_es.py` initialises and mutates the Michigan-style population.
3. `selection.py` selects the hidden neurons that survive each generation.
   The default strategy is OMP-style residual selection.
4. `logistic_output.py` fits the output layer with logistic loss and L2
   regularisation.
5. `train.py` orchestrates the full loop and can save model arrays plus a JSON
   trace of metrics.

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
split and save a plot:

```bash
python compare_strategies.py --epochs 100 --output-dir runs/strategy_comparison
```

The comparison plot contains the training-accuracy curve for each strategy and
the final validation/test accuracy bars. Use validation accuracy for strategy
selection, and keep test accuracy as the final held-out estimate.

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
