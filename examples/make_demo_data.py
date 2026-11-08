"""Generate fictional data, never the author's unpublished geothermal dataset."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / "samples.csv", args.output / "prediction.csv"]
    if any(p.exists() for p in paths):
        raise FileExistsError("Demo CSVs already exist; choose a different --output directory.")
    rng = np.random.default_rng(2026)
    xx, yy = np.meshgrid(np.arange(24)*1000 + 500, np.arange(24)*1000 + 500)
    x, y = xx.ravel(), yy.ravel()
    distance = np.abs(x - (10000 + 3000*np.sin(y/5000)))
    thermal = np.sin(x/6000) + np.cos(y/8000) + rng.normal(0, .15, len(x))
    deformation = rng.normal(0, 1e-12, len(x))
    surface = 285 + 4*thermal + rng.normal(0, 2, len(x))
    depth = 25 - 3*thermal + rng.normal(0, 1, len(x))
    noise = rng.normal(size=len(x))
    score = -distance/6000 + thermal + deformation*1e12*.4
    weights = np.exp(score - score.max())
    positive = rng.choice(len(x), 65, replace=False, p=weights/weights.sum())
    frame = pd.DataFrame({"id": [f"cell_{i}" for i in range(len(x))], "x": x, "y": y,
                          "distance": distance, "thermal_proxy": thermal, "deformation": deformation,
                          "surface_temperature": surface, "depth_proxy": depth, "noise": noise})
    frame.to_csv(paths[1], index=False)
    frame["label"] = 0  # unlabeled, not confirmed negative
    frame.loc[positive, "label"] = 1
    frame.to_csv(paths[0], index=False)
    print(f"Synthetic data: {len(positive)} positives, {len(frame)-len(positive)} unlabeled; {args.output}")


if __name__ == "__main__":
    main()
