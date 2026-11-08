# Migration from RF-GA-Average-20261014.py

This project reorganizes the supplied research script; it is not a promise to recreate historical AUCs or figures exactly. Actual research datasets, paths and figure layouts are deliberately outside the reusable core.

## Preserved workflow

- Fixed known-positive pool and repeated uniform draws from the unlabeled candidate pool.
- Temporary negative labels for sampled background rows, with reuse allowed across draws.
- Parameter and positive/unlabeled ratio selection using repeated cross-validation.
- Final sampling budget computed after ratio selection using expected coverage.
- Separate final internal CV and full-fit ensemble prediction stages.
- Average prediction maps, per-draw AUCs and per-fit SHAP importance.
- Pooled SHAP dependence statistics with quantile bins, optional percentile clipping, and inclusion of the last bin's maximum.
- Optional all-training-sample SHAP interactions and ranking of interaction pairs.
- Feature correlation, descriptive capture curve, figures, tables and georeferenced raster outputs.

## Configuration mapping

| Original variable | New configuration |
|---|---|
| `DATA_DIR`, `raster_files`, `feature_names` | `data.rasters`, relative paths, feature names as keys |
| `positive_shp`, `negative_shp` | `data.positive_points`, `data.unlabeled_points` |
| `OUTPUT_DIR` | `output_dir` |
| `N_REPEATS_GA` | `optimizer.screening_rounds` |
| `COVERAGE_TARGET` | `sampling.coverage` |
| `sampleRatio` | `sampling.ratio` or its optimizer domain |
| `numTrees` | `model.n_estimators` search parameter |
| `minLeafSize` | `model.min_samples_leaf` |
| `maxNumSplits` | `model.max_leaf_nodes` (leaf-node count, not maximum tree depth) |
| `costPenalty` | `training.positive_weight` |
| `POP_SIZE`, `MAX_ITER` | `optimizer.options.population_size`, `generations` |
| `USE_GROUP_CV`, `GRID_SIZE` | `cv.strategy: spatial`, `data.block_size` |
| `STRAIN_INDICES`, `STRAIN_SCALE` | `preprocess.feature_scales` by feature name |
| `StandardScaler(with_std=False)` | `preprocess.scaling: center` |
| `batchSize` | `prediction.batch_size` |
| `N_BINS`, `CLIP_PERCENTILE` | `explain.bins`, `clip_percentile` |
| `N_TOP_INTERACTIONS` | `explain.top_interactions` |

## Intentional corrections and changes

1. Timestamp-dependent randomness is replaced by explicit seeded streams. Default classifier randomness stays fixed across draws. Search candidates share sampling seeds to reduce noise in candidate comparison.
2. Inputs are checked for finite values and correct alignment. Missing values are not filled with zero or clipped to ±1e10. Use an explicit preprocessing step before running if imputation is needed.
3. Point-to-cell lookup uses floor; points outside the raster are not clamped into border pixels. Vector CRS is checked/reprojected, duplicate grid-cell samples and P/U overlaps are rejected.
4. Expected coverage and actual observed coverage are separately reported. Explicit round overrides are labeled as such.
5. The original RF class-cost weight is implemented with per-row sample weights, allowing the same control for other compatible estimators. No automatic second class-weight mechanism is added.
6. Search, CV, prediction and explanations are independently implemented. Model selection can be turned off or switched to another optimizer without changing sampling logic.
7. Final repeated CV is identified as internal evaluation after selection, not an independent estimate of generalization. There is no added nested CV.
8. In-sample capture is explicitly named and plotted as a descriptive diagnostic; it is not presented as an additional ROC AUC.
9. Raw SHAP and interaction arrays are written per draw instead of holding every interaction tensor in RAM simultaneously. Every selected explanation draw must succeed; errors are not silently swallowed.
10. SHAP distribution and dependence figures now use the statistical definitions in manuscript Figs. 7–9; quantile-bin x positions use boundary midpoints, as in the source script. Generic figures do not reproduce publication-specific panel assembly, fonts or geographic decorations. DPI defaults to 600 and physical SHAP feature values are retained.

The research-scale raster example preserves the original search ranges and screening count, but uses placeholder feature files. Replace it with the eligible study dataset. Runtime and results depend on the dataset, model, library versions, search budget and seed.
