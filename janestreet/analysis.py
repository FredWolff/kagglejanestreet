"""Feature analysis tool for assessing predictive power against a target column.
"""

import polars as pl


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
                "t_target_to_feature", sorted by the absolute value of
                "t_feature_to_target" descending. Features for which either
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
            .with_columns(pl.col("t_feature_to_target").abs().alias("_abs_t"))
            .sort("_abs_t", descending=True)
            .drop("_abs_t")
        )
        return result

    def analyze(self) -> dict[str, pl.DataFrame]:
        """Runs the full analysis, computing every defined metric.

        Returns:
            dict[str, pl.DataFrame]: Mapping of metric name to its result table.
        """
        return {
            "correlation": self.correlation(),
            "rolling_correlation": self.rolling_correlation(),
            "information_transfer": self.information_transfer(),
        }
