import argparse
import json
from pathlib import Path
import pandas as pd
from .config import load_config
from .pipeline import run, validate
from .ensemble import predict_saved
from .replot import replot


def main():
    parser = argparse.ArgumentParser(description="Configurable positive-unlabeled bagging")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate"):
        command = sub.add_parser(name)
        command.add_argument("config", type=Path)
    predict = sub.add_parser("predict", help="Only load trusted saved model directories")
    predict.add_argument("run_directory", type=Path)
    predict.add_argument("csv", type=Path)
    predict.add_argument("output", type=Path)
    plot = sub.add_parser("plot", help="Regenerate figures from exported result tables; no model refit")
    plot.add_argument("run_directory", type=Path)
    plot.add_argument("output_directory", type=Path)
    plot.add_argument("--dpi", type=int)
    plot.add_argument("--formats", nargs="+")
    plot.add_argument("--only", nargs="+", help="Plot families, e.g. auc roc shap_dependence")
    args = parser.parse_args()
    try:
        if args.command == "run":
            result = run(args.config)
        elif args.command == "validate":
            _, result = validate(load_config(args.config))
        elif args.command == "plot":
            result = replot(args.run_directory, args.output_directory, args.dpi, args.formats, args.only)
        else:
            if args.output.exists():
                raise FileExistsError(f"Output already exists: {args.output}")
            frame = pd.read_csv(args.csv)
            mean, std = predict_saved(args.run_directory, frame)
            frame["probability_mean"], frame["probability_std"] = mean, std
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            result = {"output": str(args.output), "rows": len(frame)}
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, KeyError, TypeError, ImportError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
