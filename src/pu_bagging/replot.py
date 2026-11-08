"""Replot exported results without the original data or any fitted model."""
from pathlib import Path
from types import SimpleNamespace
import json
from .plots import plot_results
from .outputs import read_tables


def replot(run_directory, output_directory, dpi=None, formats=None, only=None):
    source, output = Path(run_directory), Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Replot output directory is not empty: {output}")
    config = json.loads((source / "config_resolved.json").read_text(encoding="utf-8"))
    context = json.loads((source / "plot_context.json").read_text(encoding="utf-8"))
    tables = read_tables(source)
    settings = dict(config["plots"])
    settings["enabled"] = True
    if dpi is not None:
        if dpi < 1:
            raise ValueError("dpi must be positive.")
        settings["dpi"] = dpi
    if formats is not None:
        settings["formats"] = formats
    if only is not None:
        families = set(settings) - {"enabled", "dpi", "formats"}
        if set(only) - families:
            raise ValueError(f"Unknown plot families: {sorted(set(only)-families)}")
        for name in families:
            settings[name] = name in only
    shape = context["raster_shape"]
    data = SimpleNamespace(raster_profile=None, valid_pixels=None)
    if shape:
        data.raster_profile = {"height": shape[0], "width": shape[1]}
        data.valid_pixels = (tables["predictions"].row * shape[1] + tables["predictions"].col).to_numpy(dtype=int)
    output.mkdir(parents=True, exist_ok=True)
    plot_results(tables, data, output, settings)
    return {"output_dir": str(output / "figures"), "source_run": str(source)}
