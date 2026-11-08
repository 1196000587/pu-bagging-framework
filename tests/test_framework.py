from pathlib import Path
import copy
import json
import sys
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from pu_bagging.sampling import sample_count, rounds_for_coverage, expected_coverage, draw_indices
from pu_bagging.ensemble import RunningMoments, predict_saved
from pu_bagging.explain import quantile_statistics
from pu_bagging.optimizers import optimize
from pu_bagging.config import load_config
from pu_bagging.data import load_data
from pu_bagging.evaluation import split_folds
from pu_bagging.models import build_model, fit_classifier, positive_probability
from pu_bagging.pipeline import run
from pu_bagging.outputs import capture_curve


@pytest.fixture
def config_file(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 3))
    X[:30, 0] += 1.2
    frame = pd.DataFrame(X, columns=['a', 'b', 'c'])
    frame['id'] = [f'p{i}' for i in range(100)]
    frame['label'] = np.r_[np.ones(30, dtype=int), np.zeros(70, dtype=int)]
    frame['x'], frame['y'] = np.arange(100), np.arange(100) % 5
    frame.to_csv(tmp_path/'samples.csv', index=False)
    config = {
        'seed': 19, 'output_dir': 'out',
        'data': {'type': 'csv', 'path': 'samples.csv', 'features': ['a', 'b', 'c'], 'x_column': 'x', 'y_column': 'y'},
        'model': {'name': 'random_forest', 'params': {'n_estimators': 8, 'max_depth': 3}},
        'sampling': {'ratio': 1.5, 'coverage': .9},
        'cv': {'strategy': 'stratified', 'folds': 3},
        'optimizer': {'name': 'none'},
        'prediction': {'save_models': True, 'batch_size': 19},
    }
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return path


def test_coverage_is_minimal_and_not_a_guarantee():
    for pool, count in [(100, 1), (86473, 1030), (100, 100)]:
        rounds = rounds_for_coverage(pool, count, .9)
        assert expected_coverage(pool, count, rounds) >= .9
        if rounds > 1:
            assert expected_coverage(pool, count, rounds-1) < .9
    with pytest.raises(ValueError):
        rounds_for_coverage(10, 2, 1)
    assert sample_count(3, 20, 1.5) == 5
    assert sample_count(10, 4, 3) == 4


def test_sampling_no_replacement_and_repeatable():
    a = draw_indices(10, 80, 3, 5)
    b = draw_indices(10, 80, 3, 5)
    assert len(a) == len(np.unique(a)) == 30
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, draw_indices(10, 80, 3, 6))


def test_online_population_variance():
    values = np.random.default_rng(1).random((10, 30))
    running = RunningMoments(30)
    for row in values:
        running.update(row)
    np.testing.assert_allclose(running.mean, values.mean(axis=0))
    np.testing.assert_allclose(running.std, values.std(axis=0))


def test_bins_retain_maximum_and_constant_features():
    x = np.array([0, 0, 1, 2, 2, 2, 3, 3, 3, 3])
    t = quantile_statistics(x, np.arange(10), bins=20, clip_percentile=0)
    assert t.n.sum() == len(x)
    assert t.iloc[-1]['max'] == 9
    t = quantile_statistics(np.ones(6), np.arange(6), clip_percentile=0)
    assert len(t) == 1 and t.n.iloc[0] == 6


@pytest.mark.parametrize('name', ['random', 'grid', 'genetic', 'examples.custom_components:midpoint_then_random'])
def test_optimizers_share_space_contract(name):
    space = {'model.x': {'type': 'int', 'low': 1, 'high': 3},
             'sampling.ratio': {'type': 'categorical', 'choices': [1, 2]}}
    settings = {'name': name, 'space': space, 'options': {'n_trials': 12, 'population_size': 5, 'generations': 3}}
    objective = lambda p: -(p['model.x'] - 2)**2 - (p['sampling.ratio'] - 1)**2
    first, table = optimize(objective, settings, 42)
    second, _ = optimize(objective, settings, 42)
    assert first == second
    assert objective(first) == table.mean_cv_auc.max()
    assert table.params.nunique() == len(table)  # cache avoids duplicate fits


def test_group_folds_do_not_leak_and_fail_on_single_class():
    groups = np.repeat(np.arange(6), 6)
    y = np.tile([0, 1, 0, 1, 0, 1], 6)
    X = np.arange(36)[:, None]
    for train, test in split_folds(X, y, groups, {'strategy': 'group', 'folds': 3}, 42):
        assert set(groups[train]).isdisjoint(groups[test])
    with pytest.raises(ValueError, match='only one class'):
        split_folds(X, np.repeat([0, 1], 18), np.repeat([0, 1], 18), {'strategy': 'group', 'folds': 2}, 42)


def test_scaler_is_fitted_only_on_training_rows():
    model = build_model({'name': 'logistic_regression'}, {'scaling': 'standard', 'feature_scales': {}}, ['a'], {}, 1)
    X = np.array([[1.], [2.], [4.], [5.]])
    fit_classifier(model, X, np.array([0, 0, 1, 1]), 2)
    assert model.named_steps['scale'].mean_[0] == 3
    positive_probability(model, np.array([[1000.]]))
    assert model.named_steps['scale'].mean_[0] == 3


def test_class_order_is_not_assumed():
    class Reversed:
        classes_ = np.array([1, 0])
        def predict_proba(self, X):
            return np.tile([.8, .2], (len(X), 1))
    np.testing.assert_array_equal(positive_probability(Reversed(), np.zeros((2, 1))), [.8, .8])


def test_custom_classifier_and_weight_support():
    model = build_model({'name': 'sklearn.naive_bayes:GaussianNB'}, {'scaling': 'none', 'feature_scales': {}}, ['x'], {}, 3)
    fit_classifier(model, np.array([[0.], [1.], [3.], [4.]]), np.array([0, 0, 1, 1]), 2)
    assert positive_probability(model, np.array([[4.]]))[0] > .5


def test_csv_rejects_duplicate_ids_and_missing_values(config_file):
    config = load_config(config_file)
    frame = pd.read_csv(config_file.parent/'samples.csv')
    frame.loc[1, 'id'] = frame.loc[0, 'id']
    frame.to_csv(config_file.parent/'samples.csv', index=False)
    with pytest.raises(ValueError, match='unique'):
        load_data(config)
    frame.loc[1, 'id'] = 'new_id'
    frame.loc[0, 'a'] = np.nan
    frame.to_csv(config_file.parent/'samples.csv', index=False)
    with pytest.raises(ValueError, match='non-finite'):
        load_data(config)
    config['data']['invalid_samples'] = 'drop'
    assert load_data(config).diagnostics['dropped_samples'] == 1


def test_rounds_depend_on_selected_ratio_and_saved_predictions(config_file):
    config = load_config(config_file)
    config['optimizer'] = {'name': 'grid', 'screening_rounds': 1,
                           'space': {'sampling.ratio': {'type': 'categorical', 'choices': [2.0]}}, 'options': {}}
    first = run(config)
    expected_rounds = rounds_for_coverage(70, 60, .9)
    assert first['metrics']['draws'] == expected_rounds
    out = Path(first['output_dir'])
    t = pd.read_csv(out/'tables/predictions.csv')
    frame = pd.read_csv(config_file.parent/'samples.csv')
    mean, std = predict_saved(out, frame, batch_size=17)
    np.testing.assert_allclose(mean, t.probability_mean)
    np.testing.assert_allclose(std, t.probability_std, atol=1e-14)
    config['output_dir'] = 'out_again'
    second = run(config)
    np.testing.assert_allclose(pd.read_csv(Path(second['output_dir'])/'tables/predictions.csv').probability_mean, mean)
    assert second['metrics'] == first['metrics']
    with pytest.raises(FileExistsError):
        run(config)


def test_capture_curve_ties_are_order_independent():
    a = capture_curve(np.array([.5, .5, .2]), np.array([1, 0, 0]))
    b = capture_curve(np.array([.5, .5, .2]), np.array([0, 1, 0]))
    pd.testing.assert_frame_equal(a, b)


def test_plots_excel_and_explanations(config_file):
    pytest.importorskip('matplotlib')
    pytest.importorskip('openpyxl')
    pytest.importorskip('shap')
    config = load_config(config_file)
    config['sampling']['rounds'] = 2
    config['explain'].update(enabled=True, method='tree', interactions=True, bins=5, clip_percentile=0)
    config['plots'].update(enabled=True, dpi=90, formats=['png'])
    config['outputs']['excel'] = True
    result = run(config)
    out = Path(result['output_dir'])
    assert (out/'results.xlsx').exists()
    assert (out/'figures/shap_dependence_01_a.png').exists()
    assert (out/'figures/shap_interaction_matrix.png').exists()
    assert len(list((out/'explanations').glob('draw_*.npz'))) == 2
    dependence = pd.read_csv(out/'tables/shap_dependence.csv')
    assert all(dependence.groupby('feature').n.sum() == 2*(30+45))
    from pu_bagging.replot import replot
    replot(out, out/'replot', dpi=80, formats=['svg'], only=['auc', 'shap_dependence'])
    assert (out/'replot/figures/auc_by_draw.svg').exists()
    assert not (out/'replot/figures/cv_roc.svg').exists()


def test_optuna_optional():
    pytest.importorskip('optuna')
    settings = {'name': 'optuna', 'space': {'model.x': {'type': 'int', 'low': 1, 'high': 3}}, 'options': {'n_trials': 5}}
    best, history = optimize(lambda p: -(p['model.x']-2)**2, settings, 4)
    assert -(best['model.x']-2)**2 == history.mean_cv_auc.max()


@pytest.mark.parametrize('name,method', [('xgboost', 'tree'), ('logistic_regression', 'permutation'), ('extra_trees', 'tree')])
def test_other_models_and_shap(config_file, name, method):
    pytest.importorskip('shap')
    if name == 'xgboost':
        pytest.importorskip('xgboost')
    config = load_config(config_file)
    config['model']['name'] = name
    config['model']['params'] = {} if name == 'logistic_regression' else {'n_estimators': 8, 'max_depth': 3}
    config['sampling']['rounds'] = 1
    config['preprocess']['scaling'] = 'standard'
    config['explain'].update(enabled=True, method=method, max_samples=10, background_samples=5)
    result = run(config)
    out = Path(result['output_dir'])
    with np.load(out/'explanations/draw_00001.npz') as item:
        assert item['X'].shape == item['shap'].shape == (10, 3)
        assert np.isfinite(item['shap']).all()


def test_raster_crs_alignment_and_bounds(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import Point
    from rasterio.transform import from_origin
    transform = from_origin(0, 400, 100, 100)
    values = np.arange(16, dtype='float32').reshape(4, 4)
    values[3, 3] = -9999
    profile = {'driver': 'GTiff', 'height': 4, 'width': 4, 'count': 1, 'dtype': 'float32',
               'crs': 'EPSG:3857', 'transform': transform, 'nodata': -9999}
    with rasterio.open(tmp_path/'f.tif', 'w', **profile) as dst:
        dst.write(values, 1)
    pos = gpd.GeoDataFrame(geometry=[Point(99, 399), Point(150, 350)], crs='EPSG:3857')
    unl = gpd.GeoDataFrame(geometry=[Point(250, 350), Point(350, 350)], crs='EPSG:3857')
    pos.to_file(tmp_path/'p.gpkg')
    unl.to_crs(4326).to_file(tmp_path/'u.gpkg')  # adapter must reproject
    config = {'_base_dir': str(tmp_path), 'cv': {'strategy': 'stratified'},
              'data': {'type': 'raster', 'rasters': {'a': 'f.tif'}, 'positive_points': 'p.gpkg', 'unlabeled_points': 'u.gpkg'}}
    data = load_data(config)
    assert data.positive[0, 0] == 0  # floor cell indexing, never nearest rounding
    assert len(data.prediction) == 15
    from pu_bagging.outputs import export_geotiffs
    export_geotiffs(data, np.full(15, .4), np.full(15, .1), tmp_path)
    with rasterio.open(tmp_path/'probability_mean.tif') as src:
        assert src.transform == transform
        assert src.crs.to_epsg() == 3857
        assert src.read(1, masked=True).mask[3, 3]
    pos = gpd.GeoDataFrame(geometry=[Point(-1, 350), Point(150, 350)], crs='EPSG:3857')
    pos.to_file(tmp_path/'outside.gpkg')
    config['data']['positive_points'] = 'outside.gpkg'
    with pytest.raises(ValueError, match='outside'):
        load_data(config)
    profile['transform'] = from_origin(10, 400, 100, 100)
    with rasterio.open(tmp_path/'misaligned.tif', 'w', **profile) as dst:
        dst.write(values, 1)
    config['data']['rasters']['b'] = 'misaligned.tif'
    with pytest.raises(ValueError, match='match exactly'):
        load_data(config)
