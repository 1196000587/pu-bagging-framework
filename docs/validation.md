# Local validation record

Validated on 2026-11-08 using synthetic data. The source research dataset was not used and the original RF script was not modified.

## Automated checks

`python -m pytest -q`: **21 passed**. Coverage includes:

- Minimal expected-coverage budgets, sample-count bounds, seeded sampling without replacement.
- Online mean/population variance, quantile-bin maximum endpoints and constant features.
- GA, random, grid, Optuna and custom-optimizer contracts.
- Spatial-group separation and explicit rejection of single-class folds.
- Preprocessing fitted only on training data; classifier probability-column order.
- Custom classifier support and positive-sample weighting.
- Duplicate IDs and missing-data handling.
- Recomputing the final budget after ratio selection.
- Repeatability and prediction from saved pipelines.
- Order-independent treatment of tied scores in capture curves.
- RF, Extra Trees, XGBoost and logistic-regression SHAP paths.
- Plotting, Excel, SHAP interactions, and independent replotting from tables.
- Point CRS conversion, raster indexing, nodata, grid-misalignment errors, and georeferenced export.

Two warnings originate from a pending affine-operator deprecation in rasterio's test transform construction; they are not test failures.

Final manuscript-method review on 2026-11-08: the suite was rerun after matching SHAP distribution/dependence displays to manuscript Figs. 7–9 and preserving quantile-boundary midpoints. Result: **21 passed**. The two updated plot families were rendered from saved synthetic results and visually inspected; PNG metadata reported approximately 600 ppi. This checks workflow consistency, not numerical reproduction of the historical study. Per the author's scope, no cross-model consensus module or additional model-specific configurations were added.

## End-to-end examples

Executed successfully: `demo.yaml` (RF + GA + spatial CV), `full_outputs.yaml` (RF + complete explanations and exports), `xgboost_optuna.yaml`, and `logistic_grid.yaml`.

The full-output example produced 20 figure types in PNG and PDF. PNG metadata reported approximately 600 ppi, and all figure types were visually reviewed. The package installed in editable mode and built a wheel successfully.

Test environment: Windows, Python 3.13.15; numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, SHAP 0.52.0, XGBoost 3.4.1, Optuna 5.0.0, rasterio 1.5.1, GeoPandas 1.1.4, matplotlib 3.11.2. Other supported Python versions are configured in GitHub Actions but were not executed locally.
