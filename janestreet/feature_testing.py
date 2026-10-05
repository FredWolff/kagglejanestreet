"""Builds the feature sets used by the tests1/tests2 experiments in featuretest.py.

tests1 (SIM 0-4): pick the 16 raw features that rank highest by a given
FeatureAnalysis method, and build the same rolling/market-average variations
DataProcessor already builds for COLS_FEATURES_CORR, but for that ranked
subset instead.

tests2 (SIM 5-9): build a much larger candidate pool - rolling/market-average
variations of every raw feature, plus autofeat composite features - then rank
that pool and keep the top 125 columns.

Both tests skip any ranked candidate whose derived column(s) can't actually be
normalized (zero/null std - see PolarsTransformer and pipeline.py's zero-std
warning), taking the next-best-ranked candidate instead. See
_select_normalizable.
"""

from pathlib import Path

import polars as pl

from janestreet import utils
from janestreet.config import (
    COL_ID, COL_DATE, COL_TIME, COL_WEIGHT, COL_TARGET, COLS_RESPONDERS, PATH_MODELS,
)
from janestreet.data_processor import DataProcessor
from janestreet.analysis import FeatureAnalysis

PATH_POOL_CHECKPOINTS = PATH_MODELS / "pool_checkpoints"

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


def _pool_candidate_cols(df: pl.DataFrame, target: str) -> list[str]:
    """Derived (non-baseline) columns of a rolling/market-average-style pool,
    i.e. those a ranking should actually compete over - the raw baseline
    features, id/calendar/leakage columns, and the target itself are always
    kept regardless of ranking, so they're not candidates.
    """
    exclude = _default_exclude(target) + ["feature_time_id"] + DataProcessor.COLS_FEATURES_CAT
    return [
        c for c in df.columns
        if c not in CANDIDATE_FEATURES_INIT and c not in exclude and c not in (target, COL_ID)
    ]


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


def _select_normalizable(
    df: pl.DataFrame,
    ranking: pl.DataFrame,
    n_select: int,
    derived_cols=lambda col: [col],
) -> list[str]:
    """Walks `ranking`'s columns best-first, keeping a candidate only if every
    column `derived_cols(candidate)` names has nonzero, non-null std in `df`.

    A zero/null std means PolarsTransformer's (x-mean)/std scaling divides by
    ~0 (see pipeline.py's zero-std warning), so a feature like that can't
    actually be normalized - skip it and take the next-best-ranked candidate
    instead, rather than keeping an unusable column.
    """
    selected = []
    for col in ranking["column"].to_list():
        if len(selected) >= n_select:
            break
        if all(df[c].std() for c in derived_cols(col)):
            selected.append(col)
    return selected


def _build_and_persist_pool_a(
    method: str,
    skip_days: int,
    target: str = COL_TARGET,
    n_select: int = 125,
    checkpoint_name: str = "pool_a_survivors",
) -> tuple[pl.DataFrame, Path]:
    """Builds the existing rolling/market-average feature pool fresh, ranks it,
    narrows it to its own top-`n_select` normalizable columns, and persists
    that survivor set's data to disk.

    This exists so the full pool (~228 rolling/market-average columns, tens
    of GB) never needs to coexist in memory with a second, differently-shaped
    candidate pool (e.g. the lagged-feature pool `build_tests3_processor`
    builds). Only this already-narrowed, much smaller survivor set does.
    Narrowing to `n_select` here is lossless for any later selection of at
    most `n_select` columns overall: this pool alone can never need to
    contribute more than `n_select` candidates to it.

    Args:
        method (str): Ranking method, one of `RANKING_METHODS`.
        skip_days (int): Passed through to the scratch `DataProcessor`. Must
            match the `skip_days` used for whatever pool this gets combined
            with later, so both pools have the same row count/order and can
            be combined by position.
        target (str, optional): Target column to rank against. Defaults to
            `COL_TARGET`.
        n_select (int, optional): Number of top-ranked columns to keep.
            Defaults to 125.
        checkpoint_name (str, optional): Filename (without extension) for the
            persisted survivor set. Defaults to "pool_a_survivors".

    The persisted survivor set also carries the baseline raw features
    (`CANDIDATE_FEATURES_INIT` + "feature_time_id") that `_finalize_features`
    always keeps regardless of ranking - not just this pool's own top-ranked
    candidates - so it's self-sufficient for assembling a final training
    dataframe without a third fetch of the raw data.

    Returns:
        tuple[pl.DataFrame, Path]: This pool's own top-`n_select` ranking
            (small - just the score columns, best-first), and the path the
            survivor set's actual column data (baseline columns + this
            pool's top-`n_select` + target) was written to.
    """
    utils.create_folder(str(PATH_POOL_CHECKPOINTS))

    dp = DataProcessor(
        "pool_a_scratch", skip_days=skip_days, cols_features_corr=CANDIDATE_FEATURES_INIT
    )
    df = dp.get_train_data()

    candidate_cols = _pool_candidate_cols(df, target)
    ranking = _rank(df, method, target, candidate_cols=candidate_cols)
    top = _select_normalizable(df, ranking, n_select)

    # Baseline features (always kept regardless of ranking) plus the id/calendar/
    # weight/other-responder columns FullPipeline.fit needs directly from the
    # dataframe (not through .features) - see build_tests3_processor's final_df.
    baseline_cols = CANDIDATE_FEATURES_INIT + ["feature_time_id"]
    passthrough_cols = [COL_ID, COL_DATE, COL_TIME, COL_WEIGHT] + [
        r for r in COLS_RESPONDERS if r != target
    ]
    path = PATH_POOL_CHECKPOINTS / f"{checkpoint_name}.parquet"
    df.select(passthrough_cols + baseline_cols + top + [target]).write_parquet(path)

    return ranking.filter(pl.col("column").is_in(top)), path


def _build_lag_pool(
    method: str,
    skip_days: int,
    target: str = COL_TARGET,
    n_select: int = 125,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Builds the lagged-feature candidate pool, ranks it, and narrows it to
    its own top-`n_select` normalizable columns.

    Unlike `_build_and_persist_pool_a`, this pool's survivor set is not
    persisted to disk - it's the last pool built before
    `build_tests3_processor` combines it with pool A's already-persisted
    survivor set, so there's nothing to free memory for afterward.

    Args:
        method (str): Ranking method, one of `RANKING_METHODS`.
        skip_days (int): Passed through to the scratch `DataProcessor`. Must
            match the `skip_days` used for pool A, so both pools have the
            same row count/order and can be combined by position.
        target (str, optional): Target column to rank against. It is not itself
            lagged (see below). Defaults to `COL_TARGET`.
        n_select (int, optional): Number of top-ranked columns to keep.
            Defaults to 125.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: This pool's own top-`n_select`
            ranking (small - just the score columns, best-first), and the
            survivor set's actual column data (top-n columns + target).
    """
    # The target is deliberately NOT lagged: responders are only available a full day late at
    # inference, but a 7/14-row lag of responder_6 reaches back within the same day. That
    # leaks (weighted corr 0.60 / R2 0.36 at lag 7), whereas a legitimate >=1-day lag has ~0.
    lag_cols = list(CANDIDATE_FEATURES_INIT)
    dp = DataProcessor(
        "pool_b_scratch", skip_days=skip_days, cols_features_corr=[], cols_lags=lag_cols,
    )
    df = dp.get_train_data()

    candidate_cols = [f"{c}_lag_{n}" for c in lag_cols for n in dp.lags]
    ranking = _rank(df, method, target, candidate_cols=candidate_cols)
    top = _select_normalizable(df, ranking, n_select)

    return ranking.filter(pl.col("column").is_in(top)), df.select(top + [target])


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

    # cols_features_corr=CANDIDATE_FEATURES_INIT (rather than []) so the scout df already
    # has every raw feature's derived variants, letting _select_normalizable check them
    # below without a second, per-candidate data load.
    scout = DataProcessor(f"{name}_scout", skip_days=skip_days, cols_features_corr=CANDIDATE_FEATURES_INIT)
    scout_df = scout.get_train_data()

    ranking = _rank(scout_df, method, target, candidate_cols=CANDIDATE_FEATURES_INIT)
    top = _select_normalizable(
        scout_df, ranking, n_select,
        derived_cols=lambda col: [
            f"{col}_diff_rolling_avg_{scout.T}",
            f"{col}_rolling_std_{scout.T}",
            f"{col}_avg_per_date_time",
        ],
    )

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
    include_autofeat: bool = False,
) -> tuple[DataProcessor, pl.DataFrame]:
    """Builds the DataProcessor + df for a tests2 SIM (5: autofeat, 6-9: ranked).

    SIM 5 ("autofeat") builds a pool made purely of autofeat composite
    features (from the raw features) and ranks it by correlation. This is
    unaffected by `include_autofeat` - it's the whole point of that SIM.

    SIM 6-9 build a pool of rolling/market-average variations of every raw
    feature and rank that pool by the given method. If `include_autofeat` is
    True, autofeat composite features (from the raw features) are appended
    to the pool before ranking too.

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
        include_autofeat (bool, optional): For SIM 6-9 only (method !=
            "autofeat"), whether to also build autofeat composite features
            and add them to the ranked pool. Off by default: turned off to
            cut memory pressure in tests2, since fitting autofeat and holding
            its composite columns is expensive on top of the ~228-column
            rolling/market-average pool already being ranked. Set True to
            restore the previous behavior.

    Returns:
        tuple[DataProcessor, pl.DataFrame]: The configured DataProcessor
            (with `.features` set) and its training dataframe (raw features,
            all rolling/market-average variations, and - if `include_autofeat`
            or `method == "autofeat"` - autofeat composites).
    """
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

        if include_autofeat:
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

        candidate_cols = _pool_candidate_cols(df, target)
        ranking = _rank(df, method, target, candidate_cols=candidate_cols)

    top = _select_normalizable(df, ranking, n_select)

    dp.name = name
    dp.features = _finalize_features(top)
    dp.feature_ranking_ = ranking
    dp._save()  # noqa: SLF001 - re-save with the finalized feature list

    return dp, df


# (column, descending) to re-sort a merged ranking best-first by. Mirrors each
# RANKING_METHODS entry's own sort key/direction (analysis.py); rolling_stats
# is the one method sorted ascending (combined_rank: lower is better).
_MERGE_SORT_KEYS = {
    "correlation": ("abs_correlation", True),
    "rolling_stats": ("combined_rank", False),
    "information_trans": ("abs_t_feature_to_target", True),
    "mutual_information": ("mutual_information", True),
}


def build_tests3_processor(
    method: str,
    name: str,
    skip_days: int,
    target: str = COL_TARGET,
    n_select: int = 125,
) -> tuple[DataProcessor, pl.DataFrame]:
    """Builds the DataProcessor + df for a tests3 SIM (lagged-feature pool).

    Ranks the existing rolling/market-average pool ("pool A") and the new
    lagged-feature pool ("pool B", see `_build_lag_pool`) independently, then
    merges their two (already-narrowed) rankings into one global
    top-`n_select`. Pool A is narrowed to its own top-`n_select` and
    persisted to disk (`_build_and_persist_pool_a`) before pool B is even
    built, so the two pools - each tens of GB - never coexist fully in
    memory; only pool A's much smaller, already-narrowed survivor set does.

    Args:
        method (str): One of "correlation", "rolling_stats",
            "information_trans", "mutual_information".
        name (str): Name for the resulting DataProcessor.
        skip_days (int): Passed through to both pools; must be the same for
            both so their dataframes have matching row count/order and can
            be combined by position.
        target (str, optional): Target column to rank features against.
            Defaults to COL_TARGET.
        n_select (int, optional): Number of top-ranked columns to keep in
            the final, merged selection. Defaults to 125.

    Returns:
        tuple[DataProcessor, pl.DataFrame]: The configured DataProcessor
            (with `.features` and `.feature_ranking_` set, matching the
            `build_tests1_processor`/`build_tests2_processor` convention)
            and its training dataframe (baseline raw features, the id/
            calendar/weight/other-responder columns FullPipeline.fit needs
            directly, and the merged top-`n_select` columns, drawn from
            whichever of pool A/pool B each one actually came from).

    Note:
        For ranking/training experiments only - not submission-ready when
        `.features` ends up including any lagged column (likely, given
        lagged raw features tend to rank well). This `DataProcessor`'s
        `cols_lags` is unset, and lagged features aren't computable at
        single-row inference time regardless (see `_get_lags`), so
        `process_test_data` raises rather than silently producing a model
        that can never actually be used for submission.
    """
    ranking_a, path_a = _build_and_persist_pool_a(
        method, skip_days, target=target, n_select=n_select,
        checkpoint_name=f"pool_a_survivors_{name}",
    )
    ranking_b, survivors_b = _build_lag_pool(method, skip_days, target=target, n_select=n_select)
    survivors_a = pl.read_parquet(path_a)
    # The checkpoint is only a memory-relief hop within this call and each SIM
    # writes its own file (tens of GB). Leaving it behind fills the disk
    # across SIMs ("No space left on device").
    path_a.unlink(missing_ok=True)

    score_col, descending = _MERGE_SORT_KEYS[method]
    merged_ranking = (
        pl.concat([ranking_a, ranking_b], how="vertical")
        .sort(score_col, descending=descending)
        .head(n_select)
    )
    top = merged_ranking["column"].to_list()

    baseline_cols = CANDIDATE_FEATURES_INIT + ["feature_time_id"]
    passthrough_cols = [COL_ID, COL_DATE, COL_TIME, COL_WEIGHT] + [
        r for r in COLS_RESPONDERS if r != target
    ]
    cols_from_a = [c for c in top if c in survivors_a.columns]
    cols_from_b = [c for c in top if c in survivors_b.columns]

    final_df = pl.concat(
        [
            survivors_a.select(passthrough_cols + baseline_cols + cols_from_a + [target]),
            survivors_b.select(cols_from_b),
        ],
        how="horizontal",
    )

    dp = DataProcessor(f"{name}_pool", skip_days=skip_days, cols_features_corr=[])
    dp.name = name
    dp.features = _finalize_features(top)
    dp.feature_ranking_ = merged_ranking
    dp._save()  # noqa: SLF001

    return dp, final_df
