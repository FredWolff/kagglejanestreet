"""Builds the feature sets used by the tests1/tests2 experiments in featuretest.py.

tests1 (SIM 0-4): pick the 16 raw features that rank highest by a given
FeatureAnalysis method, and build the same rolling/market-average variations
DataProcessor already builds for COLS_FEATURES_CORR, but for that ranked
subset instead.

tests2 (SIM 5-9): build a much larger candidate pool - rolling/market-average
variations of every raw feature, plus autofeat composite features - then rank
that pool and keep the top 125 columns.
"""

import polars as pl

from janestreet.config import COL_ID, COL_DATE, COL_TIME, COL_WEIGHT, COL_TARGET, COLS_RESPONDERS
from janestreet.data_processor import DataProcessor
from janestreet.analysis import FeatureAnalysis

# tests1/tests2 method names -> FeatureAnalysis ranking call. "rolling_stats"
# maps to rolling_correlation, the only rolling-window ranking method available.
RANKING_METHODS = {
    "correlation": lambda fa: fa.correlation(),
    "rolling_stats": lambda fa: fa.rolling_correlation(),
    "information_trans": lambda fa: fa.information_transfer(),
    "mutual_information": lambda fa: fa.mutual_information(),
}

CANDIDATE_FEATURES_INIT = [
    c for c in DataProcessor.COLS_FEATURES_INIT if c not in DataProcessor.COLS_FEATURES_CAT
]


def _default_exclude(target: str) -> list[str]:
    # COLS_RESPONDERS already covers responder_0..10, including the
    # responder_9/responder_10 columns DataProcessor derives at load time.
    # "row_id"/"is_scored" are raw non-feature columns carried through from
    # train.parquet (see DataProcessor._get_window_average_std's fast branch).
    return (
        [COL_DATE, COL_TIME, COL_WEIGHT, "row_id", "is_scored"]
        + [r for r in COLS_RESPONDERS if r != target]
    )


def _rank(
    df: pl.DataFrame,
    method: str,
    target: str,
    candidate_cols: list[str],
    group_by: str = COL_ID,
) -> pl.DataFrame:
    """Ranks `candidate_cols` by `method`, best-first, keeping the score columns."""
    fa = FeatureAnalysis(
        df.select(candidate_cols + [target, group_by]),
        target=target,
        group_by=group_by,
    )
    return RANKING_METHODS[method](fa)


def _finalize_features(top: list[str]) -> list[str]:
    features = list(CANDIDATE_FEATURES_INIT)
    features += [c for c in top if c not in features]
    features += ["feature_time_id"]
    return features


def build_tests1_processor(
    method: str,
    name: str,
    skip_days: int,
    target: str = COL_TARGET,
    n_select: int = 16,
) -> tuple[DataProcessor, pl.DataFrame]:
    """Builds the DataProcessor + df for a tests1 SIM (0: base, 1-4: ranked).

    Args:
        method (str): One of "base", "correlation", "rolling_stats",
            "information_trans", "mutual_information".
        name (str): Name for the resulting DataProcessor.
        skip_days (int): Passed through to DataProcessor as `skip_days`.
        target (str, optional): Target column to rank features against.
            Defaults to COL_TARGET.
        n_select (int, optional): Number of top-ranked raw features to keep.
            Defaults to 16, matching the current COLS_FEATURES_CORR size.

    Returns:
        tuple[DataProcessor, pl.DataFrame]: The configured DataProcessor
            (with `.features` set) and its training dataframe.
    """
    if method == "base":
        dp = DataProcessor(name, skip_days=skip_days)
        dp.feature_ranking_ = None
        return dp, dp.get_train_data()

    scout = DataProcessor(f"{name}_scout", skip_days=skip_days, cols_features_corr=[])
    scout_df = scout.get_train_data()

    ranking = _rank(scout_df, method, target, candidate_cols=CANDIDATE_FEATURES_INIT)
    top = ranking["column"].head(n_select).to_list()

    dp = DataProcessor(name, skip_days=skip_days, cols_features_corr=top)
    dp.feature_ranking_ = ranking
    return dp, dp.get_train_data()


def build_tests2_processor(
    method: str,
    name: str,
    skip_days: int,
    target: str = COL_TARGET,
    n_select: int = 125,
    autofeat_n_input: int = 50,
    autofeat_by: str = "correlation",
    autofeat_feateng_steps: int = 2,
    autofeat_sample_size: int = 200_000,
) -> tuple[DataProcessor, pl.DataFrame]:
    """Builds the DataProcessor + df for a tests2 SIM (5: autofeat, 6-9: ranked).

    SIM 5 ("autofeat") builds a pool made purely of autofeat composite
    features (from the raw features) and ranks it by correlation.

    SIM 6-9 build a pool of rolling/market-average variations of every raw
    feature plus autofeat composite features, and rank that combined pool by
    the given method.

    In both cases the final feature list is the 79 raw features plus the top
    `n_select` ranked columns from the pool.

    Args:
        method (str): One of "autofeat", "correlation", "rolling_stats",
            "information_trans", "mutual_information".
        name (str): Name for the resulting DataProcessor.
        skip_days (int): Passed through to DataProcessor as `skip_days`.
        target (str, optional): Target column to rank features against.
            Defaults to COL_TARGET.
        n_select (int, optional): Number of top-ranked pool columns to keep.
            Defaults to 125.
        autofeat_n_input (int, optional): Number of top-ranked raw features
            fed into autofeat (kept small since autofeat's search is
            combinatorial in this count). Defaults to 25.
        autofeat_by (str, optional): FeatureAnalysis method used to pick
            autofeat's input features. Defaults to "mutual_information".
        autofeat_feateng_steps (int, optional): Passed through to
            `FeatureAnalysis.build_composite_features`. Defaults to 2.
        autofeat_sample_size (int, optional): Rows to subsample before
            fitting autofeat (see `FeatureAnalysis.build_composite_features`'s
            `sample_size` - without this, autofeat's combinatorial search
            over tens of millions of rows exhausts memory). Defaults to
            200_000.

    Returns:
        tuple[DataProcessor, pl.DataFrame]: The configured DataProcessor
            (with `.features` set) and its training dataframe (raw features,
            all rolling/market-average variations, and autofeat composites).
    """
    exclude = _default_exclude(target) + ["feature_time_id"] + DataProcessor.COLS_FEATURES_CAT

    if method == "autofeat":
        dp = DataProcessor(f"{name}_pool", skip_days=skip_days, cols_features_corr=[])
        df = dp.get_train_data()

        fa_auto = FeatureAnalysis(
            df.select(CANDIDATE_FEATURES_INIT + [target]),
            target=target,
        )
        composite = fa_auto.build_composite_features(
            n_features=autofeat_n_input,
            by=autofeat_by,
            feateng_steps=autofeat_feateng_steps,
            full_transform=True,
            sample_size=autofeat_sample_size,
        )
        df = pl.concat([df, composite], how="horizontal")

        ranking = _rank(df, "correlation", target, candidate_cols=composite.columns)
    else:
        dp = DataProcessor(
            f"{name}_pool", skip_days=skip_days, cols_features_corr=CANDIDATE_FEATURES_INIT
        )
        df = dp.get_train_data()

        fa_auto = FeatureAnalysis(
            df.select(CANDIDATE_FEATURES_INIT + [target]),
            target=target,
        )
        composite = fa_auto.build_composite_features(
            n_features=autofeat_n_input,
            by=autofeat_by,
            feateng_steps=autofeat_feateng_steps,
            full_transform=True,
            sample_size=autofeat_sample_size,
        )
        df = pl.concat([df, composite], how="horizontal")

        candidate_cols = [
            c for c in df.columns
            if c not in CANDIDATE_FEATURES_INIT and c not in exclude and c not in (target, COL_ID)
        ]
        ranking = _rank(df, method, target, candidate_cols=candidate_cols)

    top = ranking["column"].head(n_select).to_list()

    dp.name = name
    dp.features = _finalize_features(top)
    dp.feature_ranking_ = ranking
    dp._save()  # noqa: SLF001 - re-save with the finalized feature list

    return dp, df
