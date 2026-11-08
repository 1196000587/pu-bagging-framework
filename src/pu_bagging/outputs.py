"""Output adapters; tables are also the single data source for plots."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def capture_curve(probability, known_positive):
    """Training-positive capture curve; tie groups retained, never a held-out ROC."""
    frame = pd.DataFrame({"probability": probability, "positive": known_positive})
    grouped = frame.groupby("probability", sort=True).agg(count=("positive", "size"), positive=("positive", "sum"))
    grouped = grouped.iloc[::-1]
    if not grouped.positive.sum():
        return pd.DataFrame(columns=["area_fraction", "positive_capture"])
    return pd.DataFrame({"area_fraction": np.r_[0, grouped["count"].cumsum() / len(frame)],
                         "positive_capture": np.r_[0, grouped.positive.cumsum() / grouped.positive.sum()]})


def export_tables(tables, directory, settings):
    directory = Path(directory)
    if settings["csv"]:
        folder = directory / "tables"
        folder.mkdir()
        for name, table in tables.items():
            table.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
    if settings["excel"]:
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise ImportError('Excel export requires pip install ".[excel]"') from exc
        mapping = []
        with pd.ExcelWriter(directory / "results.xlsx", engine="openpyxl") as writer:
            for index, (name, table) in enumerate(tables.items(), 1):
                if len(table.columns) > 16384:
                    raise ValueError(f"{name} exceeds the Excel column limit; use CSV.")
                for part, start in enumerate(range(0, max(1, len(table)), 1048575), 1):
                    sheet = f"T{index:02d}_{part}_{re.sub(r'[^A-Za-z0-9_]', '_', name)}"[:31]
                    table.iloc[start:start + 1048575].to_excel(writer, sheet_name=sheet, index=False)
                    mapping.append({"table": name, "sheet": sheet, "start_row_zero_based": start})
            pd.DataFrame(mapping).to_excel(writer, sheet_name="Index", index=False)


def read_tables(directory):
    """Read exported tables without the original data or fitted estimators."""
    directory = Path(directory)
    tables = {}
    csvs = sorted((directory / "tables").glob("*.csv"))
    if csvs:
        for path in csvs:
            try:
                tables[path.stem] = pd.read_csv(path, dtype={"id": str})
            except pd.errors.EmptyDataError:
                tables[path.stem] = pd.DataFrame()
    elif (directory / "results.xlsx").exists():
        with pd.ExcelFile(directory / "results.xlsx", engine="openpyxl") as workbook:
            index = pd.read_excel(workbook, sheet_name="Index")
            for name, group in index.groupby("table", sort=False):
                tables[name] = pd.concat([pd.read_excel(workbook, sheet_name=s, dtype={"id": str})
                                         for s in group.sort_values("start_row_zero_based").sheet], ignore_index=True)
    else:
        raise ValueError("Saved-result operations require exported CSV or Excel tables.")
    return tables


def export_geotiffs(data, mean, std, directory):
    if data.raster_profile is None:
        return
    import rasterio
    profile = dict(data.raster_profile)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    # Remove RGB/alpha options inherited from unusual source profiles.
    profile.pop("photometric", None)
    for name, values in [("probability_mean", mean), ("probability_std", std)]:
        image = np.full(profile["height"] * profile["width"], -9999, dtype=np.float32)
        image[data.valid_pixels] = values
        with rasterio.open(Path(directory) / f"{name}.tif", "w", **profile) as dst:
            dst.write(image.reshape(profile["height"], profile["width"]), 1)
            dst.update_tags(description=("Ensemble mean of positive-vs-unlabeled scores" if name.endswith("mean")
                                         else "Population standard deviation across fitted ensemble members; not a confidence interval"))
