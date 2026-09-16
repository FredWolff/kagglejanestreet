"""Read-side analysis of Weights & Biases training runs for this project.
"""

import tempfile
from pathlib import Path

import polars as pl
import wandb

from janestreet import config
from janestreet.analysis import FeatureAnalysis

NON_CONFIG_COLUMNS = {"run_id", "name", "state", "created_at"}
NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64, pl.Boolean,
)


class WandbRuns:
    """Pulls runs from a Weights & Biases project into a polars DataFrame
    and provides analyses on top of it.

    Attributes:
        api: Weights & Biases API client used to query runs.
        project (str): W&B project to pull runs from.
        entity (str): W&B entity (team/user) owning `project`.
        state (str or None): Run state to filter on, e.g. "finished".
        category (str or None): Run `category` config value to filter on.
    """

    def __init__(
        self,
        api: object | None = None,
        project: str | None = None,
        entity: str | None = None,
        state: str | None = "finished",
        category: str | None = None,
    ) -> None:
        """Initializes the WandbRuns reader.

        Args:
            api (optional): Weights & Biases API client. Defaults to
                `wandb.Api()`. Injectable so tests can supply a fake client.
            project (str, optional): W&B project to pull runs from.
                Defaults to `config.WANDB_PROJECT`.
            entity (str, optional): W&B entity owning `project`. Defaults to
                the API client's default entity.
            state (str, optional): Run state to filter on. Defaults to
                "finished"; pass None to include every state.
            category (str, optional): Run `category` config value to filter
                on. Defaults to None (no category filter).
        """
        self.api = api if api is not None else wandb.Api()
        self.project = project or config.WANDB_PROJECT
        self.entity = entity or self.api.default_entity
        self.state = state
        self.category = category
        self._config_columns: set[str] = set()

    def _path(self) -> str:
        return f"{self.entity}/{self.project}"

    def _filters(self) -> dict:
        filters = {}
        if self.state is not None:
            filters["state"] = self.state
        if self.category is not None:
            filters["config.category"] = self.category
        return filters

    def pull(self) -> pl.DataFrame:
        """Pulls runs matching `state`/`category` into a polars DataFrame.

        Uses a single, non-lazy query so each run's config and summary
        metrics are fetched up front, in one round trip. No per-step
        history or artifacts are fetched.

        Returns:
            pl.DataFrame: One row per run, with columns "run_id", "name",
                "state", "created_at", every flattened config field (e.g.
                "category", "comment", "lr"), and every summary metric
                (e.g. "fold_0", "cv").
        """
        runs = self.api.runs(self._path(), filters=self._filters(), lazy=False)

        rows = []
        config_columns: set[str] = set()
        for run in runs:
            flat_config = self._flatten(dict(run.config))
            config_columns.update(flat_config.keys())

            # Config/summary applied first, identity fields last, so a
            # user-controlled config or summary key (e.g. a summary metric
            # named "name") can never clobber the run's actual identity.
            row = {}
            row.update(flat_config)
            row.update(dict(run.summary))
            row.update({
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
            })
            rows.append(row)

        self._config_columns = config_columns
        if not rows:
            return pl.DataFrame(
                schema={"run_id": pl.Utf8, "name": pl.Utf8, "state": pl.Utf8, "created_at": pl.Utf8}
            )
        return pl.DataFrame(rows)

    @staticmethod
    def _flatten(fields: dict) -> dict:
        """Expands list-valued fields (e.g. `hidden_sizes`) into indexed
        columns (`hidden_sizes_0`, `hidden_sizes_1`, ...) so every value is
        scalar and usable in correlation/ranking.
        """
        flat = {}
        for key, value in fields.items():
            if isinstance(value, list):
                for i, item in enumerate(value):
                    flat[f"{key}_{i}"] = item
            else:
                flat[key] = value
        return flat

    def leaderboard(
        self,
        category: str | None = None,
        comment: str | None = None,
        df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Ranks pulled runs by `cv` score.

        Args:
            category (str, optional): Only include runs with this `category`.
            comment (str, optional): Only include runs with this `comment`.
            df (pl.DataFrame, optional): Already-pulled runs, as returned by
                `pull()`. Defaults to a fresh `pull()` call; `analyze()`
                passes its own single pull in here so every view shares it.

        Returns:
            pl.DataFrame: The pulled runs, optionally filtered, sorted by
                "cv" descending. Empty (with no "cv" column added) if no
                runs were pulled.
        """
        result = df if df is not None else self.pull()
        if category is not None and "category" in result.columns:
            result = result.filter(pl.col("category") == category)
        if comment is not None and "comment" in result.columns:
            result = result.filter(pl.col("comment") == comment)
        if "cv" not in result.columns:
            return result
        return result.sort("cv", descending=True)

    def _numeric_config_columns(self, df: pl.DataFrame) -> list[str]:
        """Config columns (as tracked by the last `pull()`) that are numeric,
        so safe to correlate. Summary metrics (e.g. `cv`, `fold_i`, or any
        other logged result like `best_iteration`) are never config fields,
        so they're excluded regardless of name.
        """
        return [
            column for column, dtype in df.schema.items()
            if column in self._config_columns
            and column not in NON_CONFIG_COLUMNS
            and dtype in NUMERIC_DTYPES
        ]

    def correlation(self, df: pl.DataFrame | None = None) -> pl.DataFrame:
        """Correlates numeric config fields (hyperparameters) with `cv`.

        Reuses `FeatureAnalysis.correlation()`'s logic: sorted by absolute
        correlation descending, undefined (NaN) correlations dropped.

        Args:
            df (pl.DataFrame, optional): Already-pulled runs. Defaults to a
                fresh `pull()` call; see `leaderboard()`.

        Returns:
            pl.DataFrame: Columns "column", "correlation", "abs_correlation".
                Empty if no runs were pulled.
        """
        df = df if df is not None else self.pull()
        if "cv" not in df.columns:
            return pl.DataFrame(
                schema={"column": pl.Utf8, "correlation": pl.Float64, "abs_correlation": pl.Float64}
            )
        numeric_config = self._numeric_config_columns(df)
        analysis = FeatureAnalysis(
            df.select(numeric_config + ["cv"]),
            target="cv",
            group_by=None,
        )
        return analysis.correlation()

    def fold_stability(self, df: pl.DataFrame | None = None) -> pl.DataFrame:
        """Ranks runs by how stable their score is across CV folds.

        For each run, computes the mean and standard deviation of its
        `fold_0..fold_n` summary metrics, then ranks runs by the sum of
        their rank in mean (absolute value, descending) and their rank in
        standard deviation (ascending, i.e. more stable is better) -- the
        same combined-rank approach as `FeatureAnalysis.rolling_correlation()`,
        applied across a run's folds instead of across a rolling window.

        Args:
            df (pl.DataFrame, optional): Already-pulled runs. Defaults to a
                fresh `pull()` call; see `leaderboard()`.

        Returns:
            pl.DataFrame: Columns "run_id", "mean_cv", "std_cv",
                "combined_rank", sorted by "combined_rank" ascending. Empty
                if no runs were pulled, or none logged any `fold_*` metric.
                Runs with an undefined mean or std (e.g. a single fold) are
                dropped.
        """
        df = df if df is not None else self.pull()
        fold_columns = [c for c in df.columns if c.startswith("fold_")]
        if not fold_columns:
            return pl.DataFrame(
                schema={
                    "run_id": pl.Utf8, "mean_cv": pl.Float64,
                    "std_cv": pl.Float64, "combined_rank": pl.Float64,
                }
            )
        folds = pl.concat_list(fold_columns)

        result = (
            df.select(
                pl.col("run_id"),
                folds.list.mean().alias("mean_cv"),
                folds.list.std().alias("std_cv"),
            )
            .filter(pl.col("mean_cv").is_not_nan() & pl.col("std_cv").is_not_nan())
            .with_columns(
                pl.col("mean_cv").abs().rank(descending=True).alias("mean_rank"),
                pl.col("std_cv").rank(descending=False).alias("std_rank"),
            )
            .with_columns((pl.col("mean_rank") + pl.col("std_rank")).alias("combined_rank"))
            .drop("mean_rank", "std_rank")
            .sort("combined_rank")
        )
        return result

    def category_comparison(self, df: pl.DataFrame | None = None) -> pl.DataFrame:
        """Compares `cv` score across `category` groups.

        Args:
            df (pl.DataFrame, optional): Already-pulled runs. Defaults to a
                fresh `pull()` call; see `leaderboard()`.

        Returns:
            pl.DataFrame: Columns "category", "best_cv", "median_cv", one
                row per category, sorted by "best_cv" descending. Empty if
                no runs were pulled.
        """
        df = df if df is not None else self.pull()
        if "category" not in df.columns or "cv" not in df.columns:
            return pl.DataFrame(
                schema={"category": pl.Utf8, "best_cv": pl.Float64, "median_cv": pl.Float64}
            )
        return (
            df.group_by("category")
            .agg(
                pl.col("cv").max().alias("best_cv"),
                pl.col("cv").median().alias("median_cv"),
            )
            .sort("best_cv", descending=True)
        )

    def analyze(self) -> dict[str, pl.DataFrame]:
        """Runs every defined view over a single pull of the runs.

        Never fetches per-run artifacts/files -- that stays behind the
        explicit, separate `get_run_features()` call.

        Returns:
            dict[str, pl.DataFrame]: Mapping of view name ("leaderboard",
                "correlation", "fold_stability", "category_comparison") to
                its result table.
        """
        df = self.pull()
        return {
            "leaderboard": self.leaderboard(df=df),
            "correlation": self.correlation(df=df),
            "fold_stability": self.fold_stability(df=df),
            "category_comparison": self.category_comparison(df=df),
        }

    def get_run_features(self, run_id: str, tmp_dir: str | Path | None = None) -> list[str]:
        """Fetches the feature list a single run actually used.

        Explicit, separate call: downloads the run's saved "feature-list"
        dataset artifact (as logged by `WandbTracker.save_features`) and
        reads its "features.txt". Never invoked automatically by `pull()`,
        any of the views above, or `analyze()`.

        Args:
            run_id (str): The run's W&B id.
            tmp_dir (str or Path, optional): Directory to download the
                artifact into. Defaults to a fresh temporary directory.

        Returns:
            list[str]: Feature names, in the order they were saved.
        """
        run = self.api.run(f"{self._path()}/{run_id}")
        artifact = next(a for a in run.logged_artifacts() if a.type == "dataset")
        download_dir = artifact.download(root=tmp_dir or tempfile.mkdtemp())
        features_file = Path(download_dir) / "features.txt"
        return [line.strip() for line in features_file.read_text().splitlines() if line.strip()]
