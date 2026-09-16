"""Feature analysis tool for assessing predictive power against a target column.
"""

import inspect
import json
from pathlib import Path

import pandas as pd
import polars as pl
import autofeat.autofeat as _autofeat_module
from autofeat import AutoFeatRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.utils import check_array as _sklearn_check_array

from .config import PATH_FEATURE_SETS


class FeatureAnalysis:
    """Analyze features in a dataset for predictive power against a target column.

    The analysis is split into one method per metric (e.g. `correlation`,
    `rolling_correlation`). `analyze` runs all of them and returns their results
    together, keyed by metric name.

    Attributes:
        df (pl.DataFrame): Dataset to analyze.
        target (str): Target column the features are analyzed against.
        group_by (str or None): Column defining independent series (e.g. symbol_id)
            so that rolling metrics don't mix unrelated series.
        features (list[str]): Feature columns considered by the analysis, i.e. all
            columns of `df` except the target, `group_by`, and `exclude`.
    """

    ROLLING_WINDOW = 10000

    def __init__(
        self,
        df: pl.DataFrame,
        target: str,
        exclude: list[str] | None = None,
        group_by: str | None = "symbol_id",
    ) -> None:
        """Initializes the FeatureAnalysis.

        Args:
            df (pl.DataFrame): Dataset to analyze.
            target (str): Target column to analyze features against.
            exclude (list[str], optional): Columns to exclude from the analysis,
                e.g. the time axis or id columns. Defaults to None.
            group_by (str, optional): Column defining independent series for
                metrics that need one (e.g. symbol_id for rolling correlation).
                Excluded from `features` and left out of the analysis if None.
                Defaults to "symbol_id".
        """
        self.df = df
        self.target = target
        self.group_by = group_by

        excluded = set(exclude or []) | {target}
        if group_by is not None:
            excluded.add(group_by)
        self.features = [c for c in df.columns if c not in excluded]

    def correlation(self) -> pl.DataFrame:
        """Computes the correlation of each feature with the target.

        Returns:
            pl.DataFrame: Columns "column", "correlation", "abs_correlation",
                sorted by "abs_correlation" descending. Features with an
                undefined (NaN) correlation, e.g. constant columns, are dropped.
        """
        cols = self.features + [self.target]
        corr = self.df.select(cols).corr()

        result = (
            pl.DataFrame({"column": cols, "correlation": corr[self.target]})
            .filter(pl.col("column") != self.target)
            .with_columns(pl.col("correlation").abs().alias("abs_correlation"))
            .filter(pl.col("abs_correlation").is_not_nan())
            .sort("abs_correlation", descending=True)
        )
        return result

    def rolling_correlation(self, window_size: int = ROLLING_WINDOW) -> pl.DataFrame:
        """Computes the rolling correlation of each feature with the target.

        The rolling correlation uses a fixed window of `window_size` points and,
        if `group_by` is set, is computed independently within each group (e.g.
        per symbol_id) so windows never span across unrelated series.

        Features are ranked by the sum of their rank in mean rolling correlation
        (absolute value, descending) and their rank in standard deviation
        (ascending, i.e. more stable is better), then sorted by that combined
        rank ascending.

        Args:
            window_size (int, optional): Rolling window size, in points.
                Defaults to `ROLLING_WINDOW` (300).

        Returns:
            pl.DataFrame: Columns "column", "mean_correlation", "std_correlation",
                "combined_rank", sorted by "combined_rank" ascending. Features
                with an undefined mean or std, e.g. constant columns, are dropped.
        """
        def rolling_corr(feature: str) -> pl.Expr:
            expr = pl.rolling_corr(pl.col(feature), pl.col(self.target), window_size=window_size)
            if self.group_by is not None:
                expr = expr.over(self.group_by)
            return expr.alias(feature)

        rolling = self.df.select([rolling_corr(feature) for feature in self.features])

        means = rolling.select([pl.col(feature).mean() for feature in self.features]).row(0)
        stds = rolling.select([pl.col(feature).std() for feature in self.features]).row(0)

        result = (
            pl.DataFrame({
                "column": self.features,
                "mean_correlation": means,
                "std_correlation": stds,
            })
            .filter(
                pl.col("mean_correlation").is_not_nan()
                & pl.col("std_correlation").is_not_nan()
            )
            .with_columns(
                pl.col("mean_correlation").abs().rank(descending=True).alias("mean_rank"),
                pl.col("std_correlation").rank(descending=False).alias("std_rank"),
            )
            .with_columns((pl.col("mean_rank") + pl.col("std_rank")).alias("combined_rank"))
            .drop("mean_rank", "std_rank")
            .sort("combined_rank")
        )
        return result

    def build_rolling_features(
        self,
        window_size: int = ROLLING_WINDOW,
        market_avg_over: list[str] | None = None,
    ) -> pl.DataFrame:
        """Adds rolling and market-average features for each feature column.

        Follows the same feature engineering as `DataProcessor`
        (`_get_window_average_std`, `_get_market_average`): for each feature X,
        computes

        - "{X}_diff_rolling_avg_{window_size}": X minus its own rolling mean
          over `window_size` points. Kept as a deviation from the rolling
          average rather than the average itself, matching
          `DataProcessor._get_window_average_std`.
        - "{X}_rolling_std_{window_size}": rolling standard deviation over the
          same window.
        - "{X}_avg_per_{market_avg_over}": the market average, i.e. the mean
          of X across all rows sharing the same `market_avg_over` values
          (e.g. all symbols at a given date_id/time_id), matching
          `DataProcessor._get_market_average`.

        If `group_by` is set, the rolling mean/std are computed within each
        group (e.g. per symbol_id) so windows never span across unrelated
        series, same as `rolling_correlation`.

        This method does not mutate `self.df`; it returns an augmented copy.
        Wrap the result in a new `FeatureAnalysis` to run `correlation`,
        `mutual_information`, or `build_composite_features` over the combined
        set of original and rolling/market-average features.

        Args:
            window_size (int, optional): Rolling window size, in points.
                Defaults to `ROLLING_WINDOW`.
            market_avg_over (list[str], optional): Columns defining the groups
                market averages are computed over. Defaults to
                ["date_id", "time_id"], matching `DataProcessor`.

        Returns:
            pl.DataFrame: `self.df` with the new rolling and market-average
                columns appended.
        """
        market_avg_over = (
            market_avg_over if market_avg_over is not None else ["date_id", "time_id"]
        )

        def rolling_exprs(col: str) -> list[pl.Expr]:
            mean_expr = pl.col(col).rolling_mean(window_size=window_size)
            std_expr = pl.col(col).rolling_std(window_size=window_size)
            if self.group_by is not None:
                mean_expr = mean_expr.over(self.group_by)
                std_expr = std_expr.over(self.group_by)
            return [
                (pl.col(col) - mean_expr).alias(f"{col}_diff_rolling_avg_{window_size}"),
                std_expr.alias(f"{col}_rolling_std_{window_size}"),
            ]

        def market_avg_expr(col: str) -> pl.Expr:
            name = f"{col}_avg_per_{'_'.join(market_avg_over)}"
            return pl.col(col).mean().over(market_avg_over).alias(name)

        new_cols = []
        for f in self.features:
            new_cols.extend(rolling_exprs(f))
            new_cols.append(market_avg_expr(f))

        return self.df.with_columns(new_cols)

    def information_transfer(self) -> pl.DataFrame:
        """Computes Liang's rate of information transfer between each feature and the target.

        Implements the bivariate maximum likelihood estimator of Liang (2014), the
        method Qian et al. (2023, https://doi.org/10.1029/2022EA002722) build on
        (in its multivariate form) to pick causal drivers before feeding an LSTM.
        For two series X and Y, the rate of information flowing from X to Y is:

            T_{X->Y} = (Cyy*Cxy*Cx,dy - Cxy^2*Cy,dy) / (Cyy^2*Cxx - Cyy*Cxy^2)

        where C_ab is the sample covariance between a and b, and C_a,db is the
        sample covariance between a and the Euler forward difference of b
        (db_n = b_{n+1} - b_n, i.e. one step ahead). Information transfer is
        directional and asymmetric, so both feature->target and target->feature
        are computed; a value significantly different from zero indicates the
        first series is causal to the second, with |T| giving the magnitude.

        If `group_by` is set, forward differences are computed within each group
        so they never span across unrelated series (e.g. symbol_id), matching
        `rolling_correlation`. Covariances are then computed globally across all
        groups, mirroring `correlation`.

        Returns:
            pl.DataFrame: Columns "column", "t_feature_to_target",
                "t_target_to_feature", "abs_t_feature_to_target", sorted by
                "abs_t_feature_to_target" descending. Features for which either
                estimate is undefined (e.g. constant columns) are dropped.
        """
        def forward_diff(col: str) -> pl.Expr:
            expr = pl.col(col).shift(-1) - pl.col(col)
            if self.group_by is not None:
                expr = expr.over(self.group_by)
            return expr.alias(f"_d_{col}")

        d_target = f"_d_{self.target}"
        frame = self.df.select(
            [pl.col(self.target), forward_diff(self.target)]
            + [pl.col(f) for f in self.features]
            + [forward_diff(f) for f in self.features]
        )

        stats = frame.select(
            pl.cov(self.target, self.target).alias("_c_yy"),
            pl.cov(self.target, d_target).alias("_c_y_dy"),
            *[pl.cov(f, f).alias(f"_c_xx_{f}") for f in self.features],
            *[pl.cov(self.target, f).alias(f"_c_xy_{f}") for f in self.features],
            *[pl.cov(f, d_target).alias(f"_c_x_dy_{f}") for f in self.features],
            *[pl.cov(self.target, f"_d_{f}").alias(f"_c_y_dx_{f}") for f in self.features],
            *[pl.cov(f, f"_d_{f}").alias(f"_c_x_dx_{f}") for f in self.features],
        ).row(0, named=True)

        c_yy = stats["_c_yy"]
        c_y_dy = stats["_c_y_dy"]

        rows = []
        for f in self.features:
            c_xx = stats[f"_c_xx_{f}"]
            c_xy = stats[f"_c_xy_{f}"]
            c_x_dy = stats[f"_c_x_dy_{f}"]
            c_y_dx = stats[f"_c_y_dx_{f}"]
            c_x_dx = stats[f"_c_x_dx_{f}"]

            denom_to_target = c_yy**2 * c_xx - c_yy * c_xy**2
            denom_to_feature = c_xx**2 * c_yy - c_xx * c_xy**2

            rows.append({
                "column": f,
                "t_feature_to_target": (
                    (c_yy * c_xy * c_x_dy - c_xy**2 * c_y_dy) / denom_to_target
                    if denom_to_target else None
                ),
                "t_target_to_feature": (
                    (c_xx * c_xy * c_y_dx - c_xy**2 * c_x_dx) / denom_to_feature
                    if denom_to_feature else None
                ),
            })

        result = (
            pl.DataFrame(rows)
            .filter(
                pl.col("t_feature_to_target").is_not_null()
                & pl.col("t_feature_to_target").is_not_nan()
                & pl.col("t_target_to_feature").is_not_null()
                & pl.col("t_target_to_feature").is_not_nan()
            )
            .with_columns(
                pl.col("t_feature_to_target").abs().alias("abs_t_feature_to_target")
            )
            .sort("abs_t_feature_to_target", descending=True)
        )
        return result

    def mutual_information(
        self, sample_size: int | None = 200_000, random_state: int = 0
    ) -> pl.DataFrame:
        """Computes the mutual information of each feature with the target.

        Uses scikit-learn's k-nearest-neighbors (KSG) estimator
        (`mutual_info_regression`), which, unlike `correlation`, also picks up
        nonlinear dependence between a feature and the target. The estimator is
        O(n log n) per feature, so rows are subsampled to `sample_size` before
        fitting.

        Each feature is evaluated independently: rows where that feature is
        null are dropped for its estimate only, so different features may use
        slightly different row sets, same as the pairwise handling in
        `correlation`.

        Args:
            sample_size (int, optional): Number of rows to subsample (without
                replacement) before estimating mutual information. Defaults to
                200_000. Pass None to use all rows.
            random_state (int, optional): Random seed for subsampling and for
                the estimator's internal noise injection. Defaults to 0.

        Returns:
            pl.DataFrame: Columns "column", "mutual_information", sorted
                descending. Mutual information is never negative, so unlike
                the other metrics there is no separate "abs_" column.
        """
        df = self.df.filter(pl.col(self.target).is_not_null())
        if sample_size is not None and df.height > sample_size:
            df = df.sample(n=sample_size, seed=random_state)

        scores = []
        for f in self.features:
            pair = df.select(f, self.target).drop_nulls(f)
            if pair.height < 2:
                continue
            x = pair[f].to_numpy().reshape(-1, 1)
            y = pair[self.target].to_numpy()
            mi = mutual_info_regression(x, y, random_state=random_state)[0]
            scores.append({"column": f, "mutual_information": mi})

        result = pl.DataFrame(
            scores, schema={"column": pl.Utf8, "mutual_information": pl.Float64}
        ).sort("mutual_information", descending=True)
        return result

    def build_composite_features(
        self,
        n_features: int,
        by: str = "correlation",
        feateng_steps: int = 2,
        featsel_runs: int = 5,
        n_jobs: int = 1,
        verbose: int = 0,
        full_transform: bool = True,
        sample_size: int | None = 200_000,
        random_state: int = 0,
    ) -> pl.DataFrame:
        """Builds composite features from the top-ranked features using autofeat.

        Ranks features with one of the other analysis methods (`by`) and feeds
        the top `n_features` of them to autofeat's `AutoFeatRegressor`, which
        generates nonlinear combinations (products, ratios, logs, powers, etc.)
        of them and keeps only the ones that improve a cross-validated linear
        fit against the target. autofeat's search is combinatorial in the
        number of input features, so keep `n_features` small (autofeat itself
        warns for input counts much above 100).

        autofeat's search materializes one dense array per combination step,
        shaped (n_fit_rows, n_candidate_columns) - with n_features=50 and
        feateng_steps=2 this is already tens of thousands of candidate columns,
        so fitting on more than a few hundred thousand rows risks exhausting
        memory well before it buys any real gain in the search's statistical
        stability (row subsampling only affects which rows autofeat's search
        sees; `full_transform` still applies the surviving formulas to every
        row of `self.df` afterward).

        Rows with a null in any selected feature or the target are dropped,
        since autofeat requires finite values.

        Args:
            n_features (int): Number of top-ranked features to use as input.
            by (str, optional): Which analysis method to rank features by; one
                of "correlation", "mutual_information", "information_transfer".
                Defaults to "mutual_information".
            feateng_steps (int, optional): Nesting depth of the composites
                (passed through to `AutoFeatRegressor`). Defaults to 2.
            featsel_runs (int, optional): Number of randomized
                feature-selection runs autofeat averages over (passed
                through). Defaults to 5.
            n_jobs (int, optional): Parallel jobs for autofeat's feature
                selection. Defaults to 1.
            verbose (int, optional): autofeat verbosity level. Defaults to 0.
            full_transform (bool, optional): If True (default), fit autofeat on
                the (sampled, null-dropped) rows as usual, but then apply the
                fitted model to every row of `self.df` (via
                `AutoFeatRegressor.transform`), so the returned frame has the
                same row count and order as `self.df` (nulls in the inputs
                propagate to nulls in the composite features, rather than
                dropping the row). This is what lets the result be
                concatenated back onto `self.df` column-wise. If False, only
                the rows used for fitting are returned, as in earlier
                versions of this method.
            sample_size (int, optional): Number of null-dropped rows to
                subsample (without replacement) before fitting autofeat, same
                purpose as `mutual_information`'s `sample_size`. Defaults to
                200_000. Pass None to fit on every null-dropped row (only
                advisable for small datasets - see the memory note above).
            random_state (int, optional): Random seed for the row subsample.
                Defaults to 0.

        Returns:
            pl.DataFrame: The newly engineered composite feature columns (not
                the original inputs). One row per row of `self.df` if
                `full_transform` is True, else one row per row kept after
                dropping nulls and subsampling. Column names are the symbolic
                expressions autofeat generated (e.g. "x1*x2"), so they double
                as a description of each feature. The formulas are also
                stored, keyed by column name, in
                `self.composite_feature_formulas_`. The fitted model is
                stored in `self.autofeat_model_` and its input feature names
                in `self.autofeat_input_features_`.
        """
        rankings = {
            "correlation": self.correlation,
            "rolling_stats": self.rolling_correlation,
            "mutual_information": self.mutual_information,
            "information_transfer": self.information_transfer,
        }
        if by not in rankings:
            raise ValueError(f"by must be one of {list(rankings)}, got {by!r}")

        top_features = rankings[by]()["column"].head(n_features).to_list()
        fit_data = self.df.select(top_features + [self.target]).drop_nulls()
        if sample_size is not None and fit_data.height > sample_size:
            fit_data = fit_data.sample(n=sample_size, seed=random_state)

        x_fit = fit_data.select(top_features).to_pandas()
        y_fit = fit_data[self.target].to_pandas()

        # autofeat 2.1.1 calls the since-removed pd.Series.ravel(); pandas >=3
        # no longer defines it, so restore a compatible shim for this call.
        if not hasattr(pd.Series, "ravel"):
            pd.Series.ravel = lambda self: self.to_numpy().ravel()

        # autofeat 2.1.1's AutoFeatRegressor.transform() calls check_array
        # with the since-renamed force_all_finite kwarg; newer sklearn only
        # accepts ensure_all_finite, so shim the module-level name autofeat
        # itself calls (not sklearn's check_array globally).
        if "force_all_finite" not in inspect.signature(_sklearn_check_array).parameters:
            def _check_array_compat(*args, force_all_finite=None, **kwargs):
                if force_all_finite is not None:
                    kwargs.setdefault("ensure_all_finite", force_all_finite)
                return _sklearn_check_array(*args, **kwargs)

            _autofeat_module.check_array = _check_array_compat

        model = AutoFeatRegressor(
            feateng_steps=feateng_steps,
            featsel_runs=featsel_runs,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        fit_transformed = model.fit_transform(x_fit, y_fit)

        self.autofeat_model_ = model
        self.autofeat_input_features_ = top_features
        self.composite_feature_formulas_ = {
            name: str(formula)
            for name, formula in model.feature_formulas_.items()
            if name in model.new_feat_cols_
        }

        if full_transform:
            transformed = model.transform(self.df.select(top_features).to_pandas())
        else:
            transformed = fit_transformed

        return pl.from_pandas(transformed[model.new_feat_cols_])

    def analyze(self) -> dict[str, pl.DataFrame]:
        """Runs the full analysis, computing every defined metric.

        Returns:
            dict[str, pl.DataFrame]: Mapping of metric name to its result table.
        """
        return {
            "correlation": self.correlation(),
            "rolling_correlation": self.rolling_correlation(),
            "information_transfer": self.information_transfer(),
            "mutual_information": self.mutual_information(),
        }


def select_top_features(
    results: dict[str, pl.DataFrame],
    n_features: int = 40,
    save_dir: str | Path | None = PATH_FEATURE_SETS,
) -> dict[str, list[str]]:
    """Selects each method's top-ranked features and saves them as candidate feature sets.

    Each value in `results` (as returned by `FeatureAnalysis.analyze`) is
    already sorted best-first by that method's own ranking criterion, so the
    top `n_features` rows of "column" are simply the features that method
    considers most useful. Saving one feature set per method lets a model be
    trained on each in turn (e.g. via `PipelineCV`) to compare which
    analysis method is actually best at picking useful features.

    Args:
        results (dict[str, pl.DataFrame]): Mapping of method name to its
            result table, as returned by `FeatureAnalysis.analyze()`. Each
            table must have a "column" column, sorted best feature first.
        n_features (int, optional): Number of top features to keep per
            method. Defaults to 40.
        save_dir (str | Path | None, optional): Directory to save one JSON
            file per method (named "{method}.json", each a plain list of
            feature names, in rank order). Pass None to skip saving.
            Defaults to `PATH_FEATURE_SETS`.

    Returns:
        dict[str, list[str]]: Mapping of method name to its top
            `n_features` feature names, in rank order.
    """
    feature_sets = {
        method: df["column"].head(n_features).to_list()
        for method, df in results.items()
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for method, features in feature_sets.items():
            with open(save_dir / f"{method}.json", "w") as f:
                json.dump(features, f, indent=2)

    return feature_sets
