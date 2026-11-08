"""Optional reusable plotting adapter. Every figure is backed by an exported table."""
import re
from pathlib import Path
import numpy as np


def plot_results(tables, data, directory, settings):
    if not settings["enabled"]:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Figures require pip install ".[plots]"') from exc
    directory = Path(directory) / "figures"
    directory.mkdir()
    formats = settings["formats"]
    if not formats or set(formats) - {"png", "pdf", "svg", "tif", "tiff"}:
        raise ValueError("plots.formats must contain png, pdf, svg, tif, or tiff.")
    def save(fig, name):
        fig.tight_layout()
        for extension in formats:
            fig.savefig(directory / f"{name}.{extension}", dpi=settings["dpi"], bbox_inches="tight")
        plt.close(fig)

    if settings["auc"]:
        values = tables["cv_draws"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(values.draw, values.mean_fold_auc, s=16, label="Draw mean fold AUC")
        ax.axhline(values.mean_fold_auc.mean(), color="tomato", label="Mean across draws")
        ax.set(xlabel="Background sampling draw", ylabel="Cross-validation ROC AUC")
        ax.legend()
        save(fig, "auc_by_draw")
    if settings["roc"]:
        t = tables["roc"]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(t.fpr, t.mean_tpr, label="Mean CV ROC")
        ax.fill_between(t.fpr, np.clip(t.mean_tpr - t.std_tpr, 0, 1),
                        np.clip(t.mean_tpr + t.std_tpr, 0, 1), alpha=.2, label="Across-draw ±1 SD")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        save(fig, "cv_roc")
    if settings["optimization"] and not tables["optimization"].empty:
        t = tables["optimization"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(t.evaluation, t.mean_cv_auc, s=12)
        ax.plot(t.evaluation, t.best_so_far, color="tomato")
        ax.set(xlabel="Unique candidate evaluation", ylabel="Screening mean CV AUC")
        save(fig, "optimization_history")
    if settings["capture"] and not tables["capture_in_sample"].empty:
        t = tables["capture_in_sample"]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(t.area_fraction, t.positive_capture)
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set(xlabel="Fraction of prediction rows ranked by score", ylabel="Known-positive capture fraction",
               title="In-sample descriptive capture curve")
        save(fig, "capture_in_sample")
    if settings["correlation"]:
        t = tables["feature_correlation"].set_index("feature")
        fig, ax = plt.subplots(figsize=(max(6, len(t)*.4), max(5, len(t)*.35)))
        im = ax.imshow(t.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(t)), t.columns, rotation=60, ha="right")
        ax.set_yticks(range(len(t)), t.index)
        fig.colorbar(im, ax=ax, label="Pearson correlation")
        save(fig, "feature_correlation")
    if settings["maps"]:
        t = tables["predictions"]
        for column in ["probability_mean", "probability_std"]:
            fig, ax = plt.subplots(figsize=(7, 5))
            if data.raster_profile:
                image = np.full(data.raster_profile["height"]*data.raster_profile["width"], np.nan)
                image[data.valid_pixels] = t[column]
                im = ax.imshow(image.reshape(data.raster_profile["height"], data.raster_profile["width"]), cmap="viridis")
                ax.set(xlabel="Raster column", ylabel="Raster row")
            elif {"x", "y"} <= set(t.columns):
                im = ax.scatter(t.x, t.y, c=t[column], s=8, cmap="viridis")
                ax.set(xlabel="x", ylabel="y")
                ax.set_aspect("equal")
            else:
                ax.plot(np.arange(len(t)), t[column], linewidth=.7)
                ax.set(xlabel="Prediction row", ylabel=column)
                im = None
            if im is not None:
                fig.colorbar(im, ax=ax, label=column)
            ax.set_title("Ensemble mean score" if column.endswith("mean") else "Variation across ensemble members (SD)")
            save(fig, column)
    if "shap_importance" not in tables:
        return
    if settings["shap_importance"]:
        t = tables["shap_importance"].sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(7, max(4, len(t)*.3)))
        ax.barh(t.feature, t.mean_abs_shap, xerr=t.std_abs_shap, capsize=2)
        ax.set_xlabel("Mean |SHAP| ± population SD across fits")
        save(fig, "shap_importance")
        each = tables["shap_importance_runs"]
        fig, ax = plt.subplots(figsize=(7, max(4, len(t)*.3)))
        # Manuscript Fig. 7: IQR box, mean marker, and mean +/- population SD.
        for position, feature in enumerate(t.feature, 1):
            values = each[feature].to_numpy()
            q25, q75 = np.quantile(values, [.25, .75])
            mean, std = values.mean(), values.std(ddof=0)
            ax.barh(position, q75-q25, left=q25, height=.6, color="steelblue", alpha=.6)
            ax.hlines(position, mean-std, mean+std, color="black", linewidth=1)
            ax.vlines([mean-std, mean+std], position-.12, position+.12, color="black", linewidth=1)
            ax.vlines(mean, position-.3, position+.3, color="black", linewidth=1.5)
        ax.set_yticks(range(1, len(t) + 1), t.feature)
        ax.set_xlabel("Per-fit mean |SHAP| (box: IQR; line: mean; whiskers: mean +/- SD)")
        save(fig, "shap_importance_distribution")

    def dependence(t, name, ylabel):
        fig, ax = plt.subplots(figsize=(7, 4))
        positions = t.center.to_numpy()
        # Manuscript Figs. 8-9: pooled bin statistics, not conventional boxplot whiskers.
        ax.fill_between(positions, t["min"], t["max"], color="lightgrey", alpha=.65)
        ax.fill_between(positions, t["q25"], t["q75"], color="steelblue", alpha=.45)
        ax.fill_between(positions, t["mean"]-t["std"], t["mean"]+t["std"], color="dimgray", alpha=.35)
        ax.plot(positions, t["mean"], "o-", color="tab:blue", markersize=3)
        ax.axhline(0, color="grey", linestyle="--", linewidth=.8)
        ax.set(xlabel=str(t.feature.iloc[0]) + " (input units)", ylabel=ylabel)
        save(fig, name)

    if settings["shap_dependence"]:
        for index, (feature, table) in enumerate(tables["shap_dependence"].groupby("feature", sort=False), 1):
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", feature)
            dependence(table, f"shap_dependence_{index:02d}_{safe}", "SHAP contribution")
    if settings["shap_interactions"] and "shap_interaction_matrix" in tables:
        matrix = tables["shap_interaction_matrix"].set_index("feature")
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap="viridis")
        ax.set_xticks(range(len(matrix)), matrix.columns, rotation=60, ha="right")
        ax.set_yticks(range(len(matrix)), matrix.index)
        fig.colorbar(im, ax=ax, label="Mean |SHAP interaction|")
        save(fig, "shap_interaction_matrix")
        if "shap_interaction_dependence" in tables:
            for rank, table in tables["shap_interaction_dependence"].groupby("rank"):
                dependence(table, f"shap_interaction_rank_{rank}", f"SHAP interaction with {table.partner.iloc[0]}")
