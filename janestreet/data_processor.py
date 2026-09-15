"""Custom data processor for feature engineering and transformation.
"""

import os
import joblib
import polars as pl

from janestreet.config import PATH_DATA, PATH_MODELS
from janestreet import utils
from janestreet.transformers import PolarsTransformer

class DataProcessor:
    """Custom data processor for feature engineering and transformation.

    This class handles loading, processing, and transforming data for training and testing.
    It includes methods for adding features such as rolling averages, standard deviations,
    and market averages.

    Attributes:
        PATH (str): Path to save and load data processors.
        COLS_FEATURES_INIT (list[str]): Initial feature columns.
        COLS_FEATURES_CORR (list[str]): Correlated feature columns for additional processing.
        COLS_FEATURES_CAT (list[str]): Categorical feature columns.
        T (int): Window size for rolling computations.
        name (str): Name of the data processor.
        skip_days (int or None): Number of days to skip when loading data.
        transformer (PolarsTransformer or None): Transformer for data preprocessing.
        features (list[str]): List of feature columns after processing.
    """
    PATH = os.path.join(PATH_MODELS, "data_processors")

    COLS_FEATURES_INIT = [f"feature_{i:02d}" for i in range(79)]

    COLS_FEATURES_CORR = [
        'feature_06',
        'feature_04',
        'feature_07',
        'feature_36',
        'feature_60',
        'feature_45',
        'feature_56',
        'feature_05',
        'feature_51',
        'feature_19',
        'feature_66',
        'feature_59',
        'feature_54',
        'feature_70',
        'feature_71', 
        'feature_72',
    ]
    COLS_FEATURES_CAT = ["feature_09", "feature_10", "feature_11"]

    T = 1000

    def __init__(
        self,
        name: str,
        skip_days: int = None,
        transformer: PolarsTransformer | None = None,
        cols_features_corr: list[str] | None = None,
        T: int | None = T,
    ):
        """Initializes the DataProcessor.

        Args:
            name (str): Name of the data processor.
            skip_days (int, optional): Number of days to skip when loading data. Defaults to None.
            transformer (PolarsTransformer, optional): Transformer for data preprocessing.
                                                       Defaults to None.
            cols_features_corr (list[str], optional): Feature columns to build rolling/market-average
                                                       variations for. Defaults to `COLS_FEATURES_CORR`.
            T (int, optional): Window size for rolling computations. Defaults to the class `T`.
        """
        self.name = name
        self.skip_days = skip_days
        self.transformer = transformer
        self.cols_features_corr = (
            list(cols_features_corr) if cols_features_corr is not None else list(self.COLS_FEATURES_CORR)
        )
        self.T = T

        self.features = list(self.COLS_FEATURES_INIT)
        self.features += [f"{i}_diff_rolling_avg_{self.T}" for i in self.cols_features_corr]
        self.features += [f"{i}_rolling_std_{self.T}" for i in self.cols_features_corr]
        self.features += [f"{i}_avg_per_date_time" for i in self.cols_features_corr]
        self.features += ["feature_time_id"]
        self.features = [i for i in self.features if i not in self.COLS_FEATURES_CAT]

        # Set by janestreet.feature_testing when cols_features_corr was picked by a
        # FeatureAnalysis ranking; the ranking table (columns + scores), for logging.
        self.feature_ranking_: pl.DataFrame | None = None

        utils.create_folder(self.PATH)

    def get_train_data(self) -> pl.DataFrame:
        """Loads, processes, and returns training data.

        Returns:
            pl.DataFrame: Processed training data.
        """
        df = self._load_data().collect()

        # Additional responders
        # (8- and 60-days moving average)
        df = df.with_columns(
            (
                pl.col("responder_8")
                + pl.col("responder_8").shift(-4).over("symbol_id")
            ).fill_null(0.0).alias("responder_9"),
            (
                pl.col("responder_6")
                + pl.col("responder_6").shift(-20).over("symbol_id")
                + pl.col("responder_6").shift(-40).over("symbol_id")
            ).fill_null(0.0).alias("responder_10"),
        )

        df = self._add_features(df)

        if self.transformer is not None:
            self.transformer.set_features(self.features)
            df = self.transformer.fit_transform(df)

        self._save()
        return df

    def process_test_data(
        self,
        df: pl.DataFrame,
        fast: bool = False,
        date_id: int = 0,
        time_id: int = 0,
        symbols: list = None
    ) -> pl.DataFrame:
        """Processes test data.

        Args:
            df (pl.DataFrame): DataFrame containing test data.
            fast (bool, optional): Whether to use fast processing mode. Defaults to False.
            date_id (int, optional): Current date id. Defaults to 0.
            time_id (int, optional): Current time id. Defaults to 0.
            symbols (list, optional): List of symbols to process. Defaults to None.

        Returns:
            pl.DataFrame: Processed test data.
        """
        df = self._add_features(df, fast=fast, date_id=date_id, time_id=time_id, symbols=symbols)
        if self.transformer is not None:
            df = self.transformer.transform(df, refit=True)
        return df

    def _save(self):
        """Saves the data processor to disk.
        """
        joblib.dump(self, f"{self.PATH}/{self.name}.joblib")

    def load(self):
        """Loads the data processor from disk.

        Returns:
            DataProcessor: Loaded data processor.
        """
        return joblib.load(f"{self.PATH}/{self.name}.joblib")

    def _load_data(self) -> pl.DataFrame:
        """Loads the training data from disk.

        Returns:
            pl.DataFrame: Loaded training data.
        """
        df = pl.scan_parquet(f'{PATH_DATA}/train.parquet')
        df = df.drop("partition_id")
        if self.skip_days is not None:
            df = df.filter(pl.col("date_id")>=self.skip_days)
        return df

    def _add_features(
        self,
        df: pl.DataFrame,
        fast: bool = False,
        date_id: int | None = None,
        time_id: int | None = None,
        symbols: list = None
    ) -> pl.DataFrame:
        """Adds features to the data.

        Args:
            df (pl.DataFrame): DataFrame to process.
            fast (bool, optional): Whether to use fast processing mode. Defaults to False.
            date_id (int, optional): Current date ID. Defaults to None.
            time_id (int, optional): Current time ID. Defaults to None.
            symbols (list, optional): List of symbols to process. Defaults to None.

        Returns:
            pl.DataFrame: DataFrame with added features.
        """
        df = self._get_window_average_std(
            df,
            self.cols_features_corr,
            n=self.T,
            fast=fast,
            date_id=date_id,
            time_id=time_id,
            symbols=symbols
        )
        df = self._get_market_average(df, self.cols_features_corr, fast=fast)

        df = df.with_columns(
            pl.col("time_id").alias("feature_time_id"),

        )
        return df

    def _get_window_average_std(
        self,
        df: pl.DataFrame,
        cols: list,
        n: int = 1000,
        fast: bool = False,
        date_id: int | None = None,
        time_id: int | None = None,
        symbols: list = None
    ) -> pl.DataFrame:
        """Computes rolling averages and standard deviations.

        Args:
            df (pl.DataFrame): DataFrame to process.
            cols (list): List of columns for which to compute rolling statistics.
            n (int, optional): Window size. Defaults to 1000.
            fast (bool, optional): Whether to use fast processing mode. Defaults to False.
                                   If True, date_id, time_id and symbols args should be set.
            date_id (int, optional): Current date ID. Defaults to None.
            time_id (int, optional): Current time ID. Defaults to None.
            symbols (list, optional): List of symbols to process. Defaults to None.

        Returns:
            pl.DataFrame: DataFrame with rolling averages and standard deviations.
        """
        if not fast:
            df = df.with_columns([
                pl.col(col).rolling_mean(window_size=n)
                .over(["symbol_id"]).alias(f"{col}_rolling_avg_{n}")
                for col in cols
            ] + [
                pl.col(col).rolling_std(window_size=n)
                .over(["symbol_id"]).alias(f"{col}_rolling_std_{n}")
                for col in cols
            ])
        else:
            df = df.group_by("symbol_id").agg([
                pl.col(col).mean().alias(f"{col}_rolling_avg_{n}")
                for col in cols
            ] + [
                pl.col(col).std().alias(f"{col}_rolling_std_{n}")
                for col in cols
            ] + [
                pl.col(col).last().alias(col)
                for col in self.COLS_FEATURES_INIT + ["row_id", "weight", "is_scored"]
            ]).filter(pl.col("symbol_id").is_in(symbols))
            df = df.with_columns(
                pl.lit(date_id).cast(pl.Int16).alias("date_id"),
                pl.lit(time_id).cast(pl.Int16).alias("time_id")
            )

        df = df.with_columns([
            (pl.col(col) - pl.col(f"{col}_rolling_avg_{n}")).alias(f"{col}_diff_rolling_avg_{n}")
            for col in cols
        ])
        df = df.drop([f"{col}_rolling_avg_{n}" for col in cols])
        return df


    def _get_market_average(
        self,
        df: pl.DataFrame,
        cols: list,
        fast: bool = False
    ) -> pl.DataFrame:
        """Computes market averages (average per date_id and time_id).

        Args:
            df (pl.DataFrame): DataFrame to process.
            cols (list): List of columns for which to compute market averages.
            fast (bool, optional): Whether to use fast processing mode. Defaults to False.

        Returns:
            pl.DataFrame: DataFrame with market averages.
        """
        if not fast:
            df = df.with_columns([
                pl.col(col)
                .mean().over(["date_id", "time_id"])
                .alias(f"{col}_avg_per_date_time")
                for col in cols
            ])
        else:
            df = df.with_columns([
                pl.col(col)
                .mean()
                .alias(f"{col}_avg_per_date_time")
                for col in cols
            ])
        return df
