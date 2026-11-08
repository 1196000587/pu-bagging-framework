# Configuration and extension contracts

Top-level sections are checked for misspellings. Every example is a complete independent YAML configuration; there is no hidden inheritance. Paths resolve from the YAML directory. A run executes one classifier; run another config to compare a second classifier.

## Model and preprocessing

`model.name` selects a built-in name or trusted `module:Class`. Models must implement sklearn's estimator and classifier interfaces (`get_params`, `set_params`, `fit`, `classes_`, `predict_proba`). Custom stochastic models should expose `random_state`; the framework controls it. Binary class order is checked rather than assuming column 1.

`model.params` uses native estimator parameter names. `n_jobs` is an estimator option; searches and sampling rounds execute sequentially. GPU models may have additional nondeterminism.

`preprocess.feature_scales` multiplies named raw features before fitting (e.g. tiny strain values by 1e12). `scaling` can be `none`, `center` (mean subtraction), or `standard` (mean and variance). The preprocessing pipeline is fitted inside each training fold and saved with its model. SHAP dependence axes retain raw physical units.

`training.positive_weight` supplies per-sample weights: positive rows receive this value and unlabeled rows receive 1. Models must accept `sample_weight` if the value differs from 1. Do not also enable `class_weight` or `scale_pos_weight` unless deliberate multiplication of weights is intended.

## Shared search space

```yaml
optimizer:
  name: random
  screening_rounds: 20
  space:
    model.max_depth: {type: int, low: 3, high: 12}
    model.learning_rate: {type: float, low: 0.01, high: 0.3, log: true}
    model.booster: {type: categorical, choices: [gbtree, dart]}
    sampling.ratio: {type: int, low: 3, high: 10}
    training.positive_weight: {type: float, low: 1, high: 10}
  options: {n_trials: 30}
```

Only include parameters supported by the selected classifier. Invalid parameters and invalid CV splits produce errors, not fabricated scores. Duplicate candidates are cached. Search uses the arithmetic mean of per-draw mean-fold AUCs (all folds weighted equally, matching the original scoring convention). Per-draw pooled out-of-fold AUC is also reported during final internal evaluation.

| Search | `options` |
|---|---|
| `none` | No search, empty space |
| `random` | `n_trials` |
| `grid` | `max_candidates`; continuous specs require `values: [...]` |
| `genetic` | `population_size`, `generations`, `crossover_probability`, `mutation_probability`, `eta_c`, `eta_m`, `patience`, `tolerance` |
| `optuna` | `n_trials`; seeded single-worker TPE |

The GA uses normalized genes, tournament selection, simulated-binary crossover, polynomial mutation, and elitist survivor selection. Integer genes are decoded by rounding, categories by index. This refactor is not a byte-for-byte reproduction of the original GA trajectory.

Custom optimizer signature:

```python
def my_optimizer(objective, space, options, seed):
    candidate = {"model.max_depth": 5, "sampling.ratio": 3}
    score = objective(candidate)  # maximize; same candidate keys as space
    return candidate             # best evaluated candidate
```

Select `optimizer.name: my_package:my_optimizer`. `examples/custom_components.py` includes a runnable example. From the repository root use `python -m pu_bagging` when importing the example module. Only trusted configurations/plugins should be run: custom imports execute Python code.

## Data and validation

CSV keys: `path`, `prediction_path` (optional), `features`, `label_column` (default `label`), `id_column` (default `id`), `group_column` for predefined group CV; `x_column`, `y_column`, `coordinate_units: projected`, `block_size`, optional `block_origin` for spatial CV.

Raster keys: ordered `rasters: {feature: file}`, `positive_points`, `unlabeled_points`, `block_size`, optional `block_origin`, `additional_nodata`. Raster nodata masks are honored. Different grids cause an error instead of being implicitly resampled. Out-of-range points are rejected rather than clamped to an edge pixel. Cell indexing uses floor. Duplicate samples within a cell and overlap between positive and unlabeled cells are rejected.

`invalid_samples: error` is the default; `drop` explicitly allows removing invalid samples and logs counts. CSV prediction features must always be finite. No automatic zero filling, undocumented clipping, or background-buffer generation is performed.

Spatial groups use `floor((coordinate-origin)/block_size)` in a projected CRS. Geographic longitude/latitude cannot be used directly with a metric block size. CSV users supply the projection themselves; raster CRS is checked. GroupKFold requires both classes in every train and validation partition. If that fails, revise block definition, folds or sampling ratio instead of ignoring invalid folds.

## Python API

```python
from pu_bagging.pipeline import run
from pu_bagging.config import load_config
from pu_bagging.ensemble import predict_saved
import pandas as pd

config = load_config("examples/demo.yaml")
config["model"]["name"] = "extra_trees"
config["output_dir"] = "../outputs/extra_trees"
result = run(config)
mean, std = predict_saved(result["output_dir"], pd.read_csv("data/demo/prediction.csv"))
```

Saved models are Python joblib artifacts; load only files you trust. They are optional and can be large. One member is loaded at a time for inference.

## Compute and plotting controls

- `sampling.max_rounds` raises an error for unexpectedly large calculated budgets; it does not silently truncate.
- `prediction.batch_size` bounds per-model prediction batches. The current raster adapter loads the aligned feature stack into memory; it is not a tile-streaming training-data engine.
- SHAP calculations run separately per member; raw arrays are stored per draw. Final quantile statistics pool one feature at a time. Interactions scale as samples × features²; disable or set `max_samples` when needed.
- `max_samples: null` explains every row in each fitted training set. A finite cap is explicitly a subsampled explanation, not the original all-training-row aggregation.
- Each plot family can be toggled independently in `plots`. `enabled: false` disables all figures while tables remain available.
- CSV/Excel toggles do not remove required metadata JSONs or logs. Plot inputs are the same in-memory tables as exports; outputs do not change with plotting settings.
- Prediction table columns are mean positive-class score and member SD. The same valid-pixel mask is used for both GeoTIFFs. Preview maps use raster row/column axes; exported GeoTIFFs preserve full georeferencing.
