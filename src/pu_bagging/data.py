"""CSV and aligned-raster inputs, with explicit identities and missing-data rules."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .config import resolve_path


@dataclass
class Dataset:
    positive: np.ndarray
    unlabeled: np.ndarray
    feature_names: list
    positive_ids: np.ndarray
    unlabeled_ids: np.ndarray
    prediction: np.ndarray
    prediction_metadata: pd.DataFrame
    positive_groups: np.ndarray | None = None
    unlabeled_groups: np.ndarray | None = None
    raster_profile: dict | None = None
    valid_pixels: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)


def spatial_groups(x, y, size, origin=(0, 0)):
    if not np.isfinite(size) or size <= 0:
        raise ValueError("data.block_size must be positive in coordinate units.")
    xy = np.column_stack([x, y]).astype(float)
    if not np.isfinite(xy).all():
        raise ValueError("Spatial coordinates must be finite.")
    cells = np.floor((xy - np.asarray(origin)) / size).astype(np.int64)
    return np.array([f"{a}:{b}" for a, b in cells])


def _features(frame, names):
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return frame[names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)


def load_csv(config):
    settings = config["data"]
    names = settings["features"]
    label, identity = settings.get("label_column", "label"), settings.get("id_column", "id")
    if label in names or identity in names:
        raise ValueError("Labels and IDs cannot be predictors.")
    frame = pd.read_csv(resolve_path(config, settings["path"]), dtype={identity: str})
    if label not in frame or identity not in frame:
        raise ValueError("CSV must include the configured label and unique ID columns.")
    if frame[identity].isna().any() or frame[identity].duplicated().any():
        raise ValueError("Sample IDs must be nonmissing and unique across both pools.")
    if frame[label].isna().any() or not set(frame[label].unique()) <= {0, 1}:
        raise ValueError("CSV labels must be 1 (positive) or 0 (unlabeled).")
    X = _features(frame, names)
    valid = np.isfinite(X).all(axis=1)
    dropped = int((~valid).sum())
    if dropped and settings.get("invalid_samples", "error") != "drop":
        raise ValueError(f"{dropped} samples have non-finite predictors; fix data or set invalid_samples: drop.")
    frame, X = frame.loc[valid].reset_index(drop=True), X[valid]
    positive = frame[label].to_numpy() == 1
    groups = None
    strategy = config["cv"]["strategy"]
    if strategy == "group":
        group = settings["group_column"]
        if frame[group].isna().any():
            raise ValueError("Group values must be nonmissing.")
        groups = frame[group].astype(str).to_numpy()
    elif strategy == "spatial":
        if settings.get("coordinate_units") != "projected":
            raise ValueError("CSV spatial CV requires coordinate_units: projected; use a consistent projected CRS.")
        groups = spatial_groups(frame[settings["x_column"]], frame[settings["y_column"]],
                                settings["block_size"], settings.get("block_origin", [0, 0]))
    pred_path = settings.get("prediction_path")
    prediction_frame = pd.read_csv(resolve_path(config, pred_path), dtype={identity: str}) if pred_path else frame.copy()
    if identity not in prediction_frame or prediction_frame[identity].isna().any() or prediction_frame[identity].duplicated().any():
        raise ValueError("Prediction rows require nonmissing unique IDs.")
    prediction = _features(prediction_frame, names)
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction features contain non-finite values; fix or pre-mask these rows.")
    columns = [identity]
    for option in ("x_column", "y_column"):
        name = settings.get(option)
        if name and name in prediction_frame and name not in columns:
            columns.append(name)
    metadata = prediction_frame[columns].copy().rename(columns={identity: "id"})
    metadata = metadata.rename(columns={settings[k]: dest for k, dest in [("x_column", "x"), ("y_column", "y")]
                                         if settings.get(k) in metadata.columns})
    ids = frame[identity].to_numpy()
    metadata["known_positive"] = metadata["id"].isin(set(ids[positive])).astype(int)
    return Dataset(X[positive], X[~positive], names, ids[positive], ids[~positive], prediction, metadata,
                   None if groups is None else groups[positive], None if groups is None else groups[~positive],
                   diagnostics={"dropped_samples": dropped, "data_type": "csv"})


def load_raster(config):
    try:
        import geopandas as gpd
        import rasterio
    except ImportError as exc:
        raise ImportError('Raster input requires pip install ".[geo]"') from exc
    settings = config["data"]
    features = settings["rasters"]
    names = list(features)
    bands, profile, crs, transform, shape = [], None, None, None, None
    for name, path in features.items():
        with rasterio.open(resolve_path(config, path)) as src:
            if src.count != 1 or src.crs is None:
                raise ValueError(f"{name}: each predictor must be single-band with a CRS.")
            if profile is None:
                profile, crs, transform, shape = src.profile.copy(), src.crs, src.transform, src.shape
            elif src.crs != crs or src.transform != transform or src.shape != shape:
                raise ValueError(f"{name}: raster CRS, transform and shape must match exactly; align predictors first.")
            values = src.read(1, masked=True).astype(float).filled(np.nan)
            for nodata in settings.get("additional_nodata", []):
                values[values == nodata] = np.nan
            bands.append(values.ravel())
    cube = np.column_stack(bands)
    valid = np.isfinite(cube).all(axis=1)
    h, w = shape
    diagnostics = {"data_type": "raster", "valid_pixels": int(valid.sum())}

    def read_points(path, kind):
        points = gpd.read_file(resolve_path(config, path))
        if points.crs is None:
            raise ValueError(f"{kind}: point CRS is missing.")
        points = points.to_crs(crs)
        if not points.geometry.geom_type.eq("Point").all() or points.geometry.is_empty.any():
            raise ValueError(f"{kind}: only nonempty Point geometries are supported.")
        x, y = points.geometry.x.to_numpy(), points.geometry.y.to_numpy()
        if not np.isfinite(np.column_stack([x, y])).all():
            raise ValueError(f"{kind}: invalid coordinates.")
        rows, cols = rasterio.transform.rowcol(transform, x, y)  # floor, never round/clamp
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        pixels = rows * w + cols
        good = inside.copy()
        good[inside] &= valid[pixels[inside]]
        diagnostics[f"{kind}_invalid_points"] = int((~good).sum())
        if not good.all() and settings.get("invalid_samples", "error") != "drop":
            raise ValueError(f"{kind}: {(~good).sum()} points are outside the raster or on nodata; no border clamping is performed.")
        pixels, x, y = pixels[good], x[good], y[good]
        if len(np.unique(pixels)) != len(pixels):
            raise ValueError(f"{kind}: multiple records fall in the same grid cell; aggregate them explicitly first.")
        return pixels, x, y

    pos, px, py = read_points(settings["positive_points"], "positive")
    unl, ux, uy = read_points(settings["unlabeled_points"], "unlabeled")
    if np.intersect1d(pos, unl).size:
        raise ValueError("Positive and unlabeled points overlap in raster cells.")
    pg = ug = None
    if config["cv"]["strategy"] == "group":
        raise ValueError("Raster input supports spatial or stratified CV; use CSV for predefined groups.")
    if config["cv"]["strategy"] == "spatial":
        if not crs.is_projected:
            raise ValueError("Spatial block CV requires a projected raster CRS.")
        size, origin = settings["block_size"], settings.get("block_origin", [0, 0])
        pg, ug = spatial_groups(px, py, size, origin), spatial_groups(ux, uy, size, origin)
    pixels = np.flatnonzero(valid)
    rows, cols = np.divmod(pixels, w)
    x, y = rasterio.transform.xy(transform, rows, cols)
    ids = np.array([f"pixel_{i}" for i in pixels])
    metadata = pd.DataFrame({"id": ids, "row": rows, "col": cols, "x": x, "y": y,
                             "known_positive": np.isin(pixels, pos).astype(int)})
    return Dataset(cube[pos], cube[unl], names, np.array([f"pixel_{i}" for i in pos]),
                   np.array([f"pixel_{i}" for i in unl]), cube[valid], metadata, pg, ug,
                   profile, pixels, diagnostics)


def load_data(config):
    kind = config["data"]["type"]
    names = config["data"].get("features", []) if kind == "csv" else list(config["data"].get("rasters", {}))
    if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Feature names must be nonempty, unique strings.")
    if {"draw", "feature"} & set(names):
        raise ValueError("Feature names 'draw' and 'feature' are reserved for output tables; rename them.")
    if kind == "csv":
        data = load_csv(config)
    elif kind == "raster":
        data = load_raster(config)
    else:
        raise ValueError("data.type must be csv or raster.")
    if len(data.feature_names) == 0 or len(set(data.feature_names)) != len(data.feature_names):
        raise ValueError("Feature names must be nonempty and unique.")
    if len(data.positive) < 2 or len(data.unlabeled) < 2 or len(data.prediction) == 0:
        raise ValueError("Need at least two positives, two unlabeled samples, and prediction rows.")
    return data
