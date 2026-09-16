"""Custom transformer for preprocessing data using Polars."""

import polars as pl

class PolarsTransformer:
    """A custom transformer for preprocessing data using Polars.

    This transformer provides functionality to scale, fill missing values, 
    and clip features in a DataFrame. It can fit on a given 
    dataset to compute statistics and apply transformations accordingly.

    Args:
        features (list, optional): List of feature columns to be transformed. Defaults to None.
        fillnull (bool, optional): Whether to fill null values with 0. Defaults to True.
        scale (bool, optional): Whether to scale the features 
                                by mean and standard deviation. Defaults to True.
        clip_time (bool, optional): Whether to clip the "feature_time_id" column 
                                    to its min and max values. Defaults to True.
    """
    def __init__(
        self,
        features: list = None,
        fillnull: bool = True,
        scale: bool = True,
        clip_time: bool = True
    ) -> None:
        """Initializes the PolarsTransformer class.

        Args:
            features (list, optional): List of feature columns to be transformed. Defaults to None.
            fillnull (bool, optional): Whether to fill null values with 0. Defaults to True.
            scale (bool, optional): Whether to scale the features 
                                    by mean and standard deviation. Defaults to True.
            clip_time (bool, optional): Whether to clip the "feature_time_id" column 
                                        to its min and max values. Defaults to True.
        """
        self.features = features
        self.fillnull = fillnull
        self.scale = scale
        self.clip_time = clip_time
        self.statistics_mean_std = None
        self.statistics_min_max = None

    def set_features(self, features: list) -> None:
        """Sets the list of features to be transformed.

        Args:
            features (list): List of feature columns to be transformed.
        """
        self.features = features

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fits the transformer on the given DataFrame and applies the transformations.

        Args:
            df (pl.DataFrame): The input Polars DataFrame.

        Returns:
            pl.DataFrame: The transformed Polars DataFrame.
        """
        if self.scale:
            self.statistics_mean_std = {
                column: {
                    "mean": df[column].mean(),
                    "std": df[column].std(),
                }
                for column in self.features
            }

        if self.clip_time:
            self.statistics_min_max = {
                column: {
                    "min": df[column].min(),
                    "max": df[column].max(),
                }
                for column in ["feature_time_id"]
            }

        if self.fillnull:
            df = df.with_columns([
                pl.col(column).fill_null(0.0)
                for column in self.features
            ])

        if self.scale:
            df = df.with_columns([
                ((pl.col(column) - (self.statistics_mean_std[column]["mean"] or 0.0)) /
                # A zero/null std means the column is (near-)constant; falling back to 1.0
                # avoids dividing by ~0, which would blow up to inf/NaN downstream.
                (self.statistics_mean_std[column]["std"] or 1.0))
                for column in self.features
            ])

        return df

    def transform(self, df: pl.DataFrame, refit: bool = False) -> pl.DataFrame:
        """Applies the transformations to the given DataFrame using precomputed statistics.

        Args:
            df (pl.DataFrame): The input Polars DataFrame.
            refit (bool, optional): If True, updates the min and max values for 
                the "feature_time_id" column. Defaults to False.

        Returns:
            pl.DataFrame: The transformed Polars DataFrame.
        """
        if refit:
            if self.clip_time:
                self.statistics_min_max.update({
                    column: {
                        "min": (
                            self.statistics_min_max[column]["min"]
                            if df[column].min() is None
                            else min(df[column].min(), self.statistics_min_max[column]["min"])
                        ),
                        "max": (
                            self.statistics_min_max[column]["max"]
                            if df[column].max() is None
                            else max(df[column].max(), self.statistics_min_max[column]["max"])
                        ),
                    }
                    for column in ["feature_time_id"]
                })

        if self.clip_time:
            df = df.with_columns([
                pl.col(column).clip(
                    self.statistics_min_max[column]["min"],
                    self.statistics_min_max[column]["max"]
                )
                for column in ["feature_time_id"]
            ])

        if self.fillnull:
            df = df.with_columns([
                pl.col(column).fill_null(0.0)
                for column in self.features
            ])

        if self.scale:
            df = df.with_columns([
                ((pl.col(column) - (self.statistics_mean_std[column]["mean"] or 0.0)) /
                    (self.statistics_mean_std[column]["std"] or 1.0))
                for column in self.features
            ])

        return df
