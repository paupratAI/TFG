# Training Neural Networks with Evolution Strategies

Bachelor's thesis (TFG) prototype: a single-hidden-layer MLP trained on
tabular data without backpropagating through the hidden layer. The hidden layer
is evolved with a Michigan-style Evolution Strategy (each individual is one
hidden neuron). The output layer is solved analytically with Ridge regression
and Generalised Cross-Validation (GCV), or trained as logistic/softmax when
the task is classification.

- Final report: [`TFG_Pau_Prat_Moreno.pdf`](TFG_Pau_Prat_Moreno.pdf)
- Defence slides: [`canva_tfg_deck_v2/index.html`](canva_tfg_deck_v2/index.html)
- LaTeX sources: [`overleaf/`](overleaf/)

## Main Flow

1. `data.py` loads a numeric OpenML dataset, splits before preprocessing,
   imputes missing values on the training split only, standardises features,
   and maps binary labels to `{-1, +1}`.
2. `core_es.py` initialises and mutates the Michigan-style population.
3. `selection.py` selects the hidden neurons that survive each generation:
   `omp`, `r2`, `cfs`, or `cfs_auc`.
4. `gcv.py` fits the output layer with an SVD-based Ridge/GCV solve.
5. `train.py` orchestrates the loop, logs accuracy/MSE/ROC-AUC/GCV/lambda, and
   optionally saves JSON/NPZ artifacts.

With a validation split, `train.py` selects the checkpoint with highest
validation accuracy, using validation MSE as tie-breaker. Test metrics are
computed only after that checkpoint has been selected.

## Quick Start

```bash
pip install -r requirements.txt
python train.py --strategy omp --data-id 59 --epochs 100 --save-dir runs
```

Run the four selection strategies:

```bash
python train.py --strategy omp --data-id 59 --epochs 100
python train.py --strategy r2 --data-id 59 --epochs 100
python train.py --strategy cfs --data-id 59 --epochs 100
python train.py --strategy cfs_auc --data-id 59 --epochs 100
```

`cfs_auc` keeps the CFS subset-merit structure, but uses squared Gini-AUC for
neuron-target relevance and squared Spearman correlation for neuron-neuron
redundancy.

## Experiments

Optionally compare the four neuron-selection strategies on a common
train/validation/test split:

```bash
python compare_strategies.py --epochs 100 --output-dir runs/strategy_comparison
```

Run the current multi-seed/multi-dataset benchmark:

```bash
python compare_strategies.py \
  --datasets ionosphere sonar breast-w \
  --seeds 42 43 44 \
  --epochs 60 \
  --mu 50 \
  --offspring 250 \
  --validation-size 0.2 \
  --output-dir runs/strategy_comparison
```

Run the defence-oriented benchmark used for the report:

```bash
python ridge_defense_suite.py \
  --datasets ionosphere sonar breast-w \
  --seeds 42 43 44 \
  --es-epochs 60 \
  --es-mu 50 \
  --es-offspring 250 \
  --output-dir runs/ridge_defense
```

This compares ES-CFS Ridge+GCV against Linear Ridge+GCV, random tanh
Ridge+GCV, Logistic Regression, RBF SVM, Random Forest, HistGradientBoosting,
and a gradient MLP. It also writes diagnostic plots and `latex_notes.tex` with
commented snippets for manual inclusion in the report.

Run the regression-oriented benchmark:

```bash
python ridge_regression_benchmark.py \
  --datasets california friedman1 diabetes \
  --data-home .sklearn_data \
  --max-samples 12000 \
  --es-strategies cfs omp r2 \
  --es-epochs 35 \
  --es-mu 60 \
  --es-offspring 180 \
  --random-hidden-grid 32 64 128 256 \
  --mlp-max-iter 120 \
  --output-dir runs/ridge_regression
```

This adapts the Ridge/GCV architecture to regression by keeping the same
evolved tanh hidden layer and solving the output layer analytically against a
continuous target. It compares Linear Ridge+GCV, random tanh Ridge+GCV,
ES-selected Ridge+GCV, HistGradientBoosting, and a gradient MLP. For the ES
prototype, `cfs`, `omp`, and `r2` are evaluated on the validation split and the
reported strategy is the one with the lowest validation RMSE.

Run the multiclass softmax benchmark:

```bash
python logistic_multiclass_benchmark.py \
  --datasets digits gaussian-quantiles spiral-3 \
  --max-samples 3000 \
  --es-strategies cfs \
  --es-epochs 55 \
  --es-mu 128 \
  --es-offspring 384 \
  --random-hidden-grid 32 64 128 \
  --output-max-iter 200 \
  --mlp-max-iter 180 \
  --output-dir runs/logistic_multiclass
```

This evaluates the probabilistic architecture for multiclass classification:
the evolved hidden representation is followed by an unregularised softmax
output layer trained with cross-entropy. It reports accuracy, macro-F1,
log-loss, and multiclass expected calibration error.

## Baselines

```bash
python benchmark_level1.py
python train_ga_conventional.py
python train_pytorch_conventional.py
```

`train_ga_conventional.py` is a Pittsburgh-style GA baseline where each
individual encodes a full hidden layer. It uses the same Ridge/GCV output
solver as the ES prototype.

## Outputs

When `--save-dir` is provided, `train.py` writes:

- `.npz`: final `alpha`, `L`, and `beta` arrays.
- `.json`: run configuration, final metrics, and per-epoch history.

OpenML cache files are written to `.openml_cache/` by default.

## Thesis and slides

The compiled bachelor's thesis is `TFG_Pau_Prat_Moreno.pdf`. Chapter sources
live in `overleaf/` (`main.txt` includes `abstract`, `introduction`, `report`,
`experiments`, and `conclusions`). Figures used in the report are in
`plots_overleaf/`.

The defence deck is a self-contained HTML presentation in `canva_tfg_deck_v2/`.
Open `canva_tfg_deck_v2/index.html` in a browser. Speaker notes are in
`canva_tfg_deck_v2/speaker-notes.txt`.
