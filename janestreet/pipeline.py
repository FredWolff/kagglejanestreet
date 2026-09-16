"""Custom pipeline classes.

These classes are designed to facilitate model training, cross-validation, and testing 
on time series data, with additional functionalities for model management.

Classes:
    FullPipeline: A custom pipeline for model training, saving, loading, and updating.
    PipelineEnsemble: An ensemble pipeline that aggregates predictions from multiple models.
    PipelineCV: A cross-validation pipeline designed for time series data.
"""

import copy
import gc
import os


import joblib
import numpy as np
import polars as pl

from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator

import torch

from .config import PATH_MODELS, COL_TARGET, COL_ID, COL_DATE, COL_TIME, COL_WEIGHT, COLS_RESPONDERS
from . import utils
from .metrics import r2_weighted
from .tracker import WandbTracker


TEST_SIZE = 200
GAP = 0


class FullPipeline:
    """Custom pipeline for model management and time series training.

    This class provides methods for fitting, predicting, updating, saving, and loading models.

    Attributes:
        model (BaseEstimator): The model to be used.
        preprocessor: Optional data preprocessing pipeline.
        name (str): Name of the model for saving/loading purposes.
        load_model (bool): Flag indicating whether to load the model from disk.
        features (list[str] or None): List of feature names to use.
        save_to_disc (bool): Flag indicating whether to save the model to disk.
        refit (bool): Flag indicating whether to refit the model during updates.
        change_lr (bool): Flag indicating whether to change learning rate (if load is True).
        col_target (str): Name of the target column.
    """
    def __init__(
            self,
            model: BaseEstimator,
            preprocessor = None,
            run_name: str = "",
            name: str = "",
            load_model: bool = False,
            features: list | None = None,
            save_to_disc: bool = True,
            refit = True,
            change_lr = False,
            col_target = COL_TARGET,
    ) -> None:
        """Initializes the FullPipeline.

        Args:
            model (BaseEstimator): The model to be used.
            preprocessor: Optional preprocessing pipeline.
            run_name (str): Name of the current run.
            name (str): Name of the model for saving/loading purposes.
            load_model (bool): Whether to load the model from disk.
            features (list[str] or None): List of feature names to use.
            save_to_disc (bool): Whether to save the model to disk.
            refit (bool): Whether to refit the model during updates.
            change_lr (bool): Whether to change learning rate (if load is True).
            col_target (str): Name of the target column.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.name = name
        self.load_model = load_model
        self.features = features
        self.save_to_disc = save_to_disc
        self.refit = refit
        self.change_lr = change_lr
        self.col_target = col_target

        self.responders = [i for i in COLS_RESPONDERS if i != self.col_target]

        self.set_run_name(run_name)
        self.path = os.path.join(PATH_MODELS, f"{self.run_name}")

    def set_run_name(self, run_name: str) -> None:
        """Sets the run name for the model.

        Args:
            run_name (str): The name of the run.

        """
        self.run_name = run_name
        self.path = os.path.join(PATH_MODELS, f"{self.run_name}")
        if self.save_to_disc:
            utils.create_folder(self.path)

    def fit(
        self,
        df: pl.DataFrame | None = None,
        df_valid: pl.DataFrame | None = None,
        verbose: bool = False,
    ) -> None:
        """Fits the model pipeline.

        Args:
            df (pl.DataFrame | None): DataFrame containing training data.
            df_valid (pl.DataFrame | None): DataFrame containing validation data.
            verbose (bool): Whether to enable verbose output during fitting.

        """
        if not self.load_model:
            self.model.features = self.features

            weights_train = df.select(COL_WEIGHT).to_series().to_numpy()
            dates_train = df.select(COL_DATE).to_series().to_numpy()
            times_train = df.select(COL_TIME).to_series().to_numpy()
            stocks_train = df.select(COL_ID).to_series().to_numpy()

            weights_valid = df_valid.select(COL_WEIGHT).to_series().to_numpy()
            dates_valid = df_valid.select(COL_DATE).to_series().to_numpy()
            times_valid = df_valid.select(COL_TIME).to_series().to_numpy()
            stocks_valid = df_valid.select(COL_ID).to_series().to_numpy()

            if self.preprocessor is not None:
                df = self.preprocessor.fit_transform(df)
                df_valid = self.preprocessor.transform(df_valid)

            X_train = df.select(self.features).to_numpy()
            resp_train = df.select(self.responders).to_numpy()
            y_train = df.select(self.col_target).to_series().to_numpy()

            X_valid = df_valid.select(self.features).to_numpy()
            resp_valid = df_valid.select(self.responders).to_numpy()
            y_valid = df_valid.select(self.col_target).to_series().to_numpy()

            if verbose:
                # A near-zero std means PolarsTransformer's (x-mean)/std scaling divides by
                # ~0, producing inf/huge values that can push the network to NaN outputs -
                # flag it here, before that happens, rather than only seeing a NaN score later.
                stats_mean_std = getattr(self.preprocessor, "statistics_mean_std", None) or {}
                zero_std = [col for col, stats in stats_mean_std.items() if not stats["std"]]
                if zero_std:
                    print(f"[{self.run_name}] WARNING: zero/undefined std features "
                          f"(scaling divides by ~0): {zero_std}")

                n_nan_train, n_inf_train = np.isnan(X_train).sum(), np.isinf(X_train).sum()
                n_nan_valid, n_inf_valid = np.isnan(X_valid).sum(), np.isinf(X_valid).sum()
                if n_nan_train or n_inf_train or n_nan_valid or n_inf_valid:
                    print(f"[{self.run_name}] WARNING: non-finite values after scaling - "
                          f"train nan={n_nan_train} inf={n_inf_train}, "
                          f"valid nan={n_nan_valid} inf={n_inf_valid}")
                print(f"[{self.run_name}] scaled feature magnitude: "
                      f"max|X_train|={np.nanmax(np.abs(X_train)):.3g}, "
                      f"max|X_valid|={np.nanmax(np.abs(X_valid)):.3g}")

            train_set = (
                X_train,
                resp_train,
                y_train,
                weights_train,
                stocks_train,
                dates_train,
                times_train
            )
            val_set = (
                X_valid,
                resp_valid,
                y_valid,
                weights_valid,
                stocks_valid,
                dates_valid,
                times_valid
            )

            del df, df_valid
            gc.collect()

            self.model.fit(train_set, val_set, verbose)
            if self.save_to_disc:
                self.save()
        else:
            self.load()

    def predict(
        self,
        df: pl.DataFrame,
        hidden: torch.Tensor | list | None = None,
        n_times: int | None = None
    ) -> tuple[np.ndarray, torch.Tensor | list]:
        """Predicts target using the fitted model.

        Args:
            df (pl.DataFrame): DataFrame containing data for prediction.
            hidden (torch.Tensor | list | None): Hidden states for recurrent models.
            n_times (int | None): Number of time steps to predict.

        Returns:
            tuple[np.ndarray, torch.Tensor | list]: Predicted probabilities and hidden states.

        """
        if n_times is None:
            n_times = len(df.select(COL_TIME).unique())
        if self.preprocessor is not None:
            df = self.preprocessor.transform(df)
        X = df.select(self.features).to_numpy()
        preds, hidden = self.model.predict(X, hidden=hidden, n_times=n_times)
        preds = np.clip(preds, -5, 5)
        return preds, hidden

    def update(self, df: pl.DataFrame) -> None:
        """Updates model weights using new data.

        Args:
            df (pl.DataFrame): DataFrame containing data for updating the model.

        """
        weights = df.select(COL_WEIGHT).to_series().to_numpy()
        n_times = len(df.select(COL_TIME).unique())
        if self.preprocessor is not None:
            df = self.preprocessor.transform(df, refit=True)

        X = df.select(self.features).to_numpy()
        y = df.select(self.col_target).to_series().to_numpy()
        self.model.update(X, y, weights, n_times)

    def load(self) -> None:
        """Loads the model from disk."""
        if self.change_lr:
            lr_refit = self.model.lr_refit
        self.model = joblib.load(f"{self.path}/model_{self.name}.joblib")
        self.features = self.model.features
        if self.change_lr:
            self.model.lr_refit = lr_refit
        try:
            self.preprocessor = joblib.load(f"{self.path}/preprocessor_{self.name}.joblib")
        except FileNotFoundError:
            self.preprocessor = None
            print("WARNING: Preprocessor not found.")

    def save(self) -> None:
        """Saves the model to disk."""

        joblib.dump(self.model, f"{self.path}/model_{self.name}.joblib")
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, f"{self.path}/preprocessor_{self.name}.joblib")

    def get_params(self, deep: bool = True) -> dict:
        """Gets parameters for the estimator.

        Args:
            deep (bool): Whether to return the parameters of sub-objects.

        Returns:
            dict: Dictionary of parameters.
        """
        return {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "name": self.name,
            "load_model": self.load_model,
            "features": self.features,
            "save_to_disc": self.save_to_disc,
            "refit": self.refit,
            "change_lr": self.change_lr,
            "col_target": self.col_target,
        }

    def set_params(self, **parameters):
        """Sets the parameters of the estimator.

        Args:
            parameters: A dictionary of parameter names and values.

        Returns:
            self: The updated estimator.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class PipelineEnsemble:
    """Ensemble pipeline for aggregating predictions from multiple models.

    This class manages multiple models, allowing for fitting, prediction, updating,
    and managing ensemble weights. It is designed for time series data, where
    different models can be combined to improve overall prediction performance.

    Attributes:
        models (list): List of models to be used in the ensemble.
        weights (np.ndarray): Array of weights for averaging predictions.
        refit_models (list[bool]): Flags indicating whether each model 
                                   should be refit during updates.
        col_target (str): Name of the target column.
    """
    def __init__(
        self,
        models: list,
        weights: np.array = None,
        refit_models: list[bool] = None,
        col_target: str = COL_TARGET
    ) -> None:
        """Initializes the PipelineEnsemble.

        Args:
            models (list): List of models to be used in the ensemble.
            weights (numpy array or None): Weights for averaging model predictions.
            refit_models (list[bool] or None): Flags for refitting models during updates.
            col_target (str): Name of the target column.
        """
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(self.models))
        self.refit_models = refit_models if refit_models is not None else [True]*len(models)
        self.col_target = col_target
        self.refit = True

    def fit(
        self,
        df: pl.DataFrame | None = None,
        df_valid: pl.DataFrame | None = None,
        verbose: bool = False,
    ) -> None:
        """Fits all models in the ensemble.

        Args:
            df (pl.DataFrame | None): DataFrame containing the training data.
            df_valid (pl.DataFrame | None): DataFrame containing the validation data.
            verbose (bool): Enables verbose output during fitting.

        """
        self.weights = np.array(self.weights) / sum(self.weights)
        for model in self.models:
            model.fit(df, df_valid, verbose)

    def set_run_name(self, run_name: str) -> None:
        """Sets the run name for all models in the ensemble.

        Args:
            run_name (str): The name of the run.

        """
        for model in self.models:
            model.set_run_name(run_name)

    def predict(self, df: pl.DataFrame, hidden_ls=None) -> np.ndarray:
        """Predicts probabilities using all models in the ensemble.

        Args:
            df (pl.DataFrame): DataFrame containing the data for prediction.
            hidden_ls (list or None): List of hidden states for each model.

        Returns:
            tuple[np.ndarray, list]: Averaged predictions and updated hidden states.
        """
        if hidden_ls is None:
            hidden_ls = [None] * len(self.models)

        preds = []
        for i, model in enumerate(self.models):
            preds_i, hidden_ls[i] = model.predict(df, hidden=hidden_ls[i])
            preds.append(preds_i)

        preds = np.average(preds, axis=0, weights=self.weights)
        return preds, hidden_ls

    def update(self, df: pl.DataFrame) -> None:
        """Updates models weights using new data.

        Args:
            df (pl.DataFrame): DataFrame containing data for updating the models.

        """
        for i, model in enumerate(self.models):
            if self.refit_models[i]:
                model.update(df)


    def load(self) -> None:
        """Loads all models in the ensemble from disk.
        """
        for model in self.models:
            model.model.load()

    def save(self) -> None:
        """Saves all models in the ensemble to disk.
        """
        for model in self.models:
            model.model.save()

    def get_params(self, deep: bool = True) -> dict:
        """Gets parameters for the ensemble.

        Args:
            deep (bool): Whether to return parameters of sub-objects.

        Returns:
            dict: Dictionary of parameters.
        """
        return {
            "models": self.models,
            "weights": self.weights,
            "refit_models": self.refit_models,
            "col_target": self.col_target,
        }

    def set_params(self, **parameters):
        """Sets the parameters of the ensemble.

        Args:
            parameters (dict): A dictionary of parameter names and values.

        Returns:
            self: The updated ensemble.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class PipelineCV:
    """Cross-validation pipeline for time series models.

    This class manages cross-validation for time series models, allowing for fitting models on
    multiple folds, tracking results, and handling time-based data splits.

    Attributes:
        model (FullPipeline): The model to be validated.
        tracker (WandbTracker): Tracker for logging metrics during cross-validation.
        n_splits (int): Number of cross-validation splits.
        train_size (int): Maximum size of the training set.
        models (list): List of models fitted on each fold.
    """
    def __init__(
            self,
            model: FullPipeline,
            tracker: WandbTracker,
            n_splits: int,
            train_size: int = False,
    ) -> None:
        """Initializes the PipelineCV.

        Args:
            model (FullPipeline): The model to be validated.
            tracker (WandbTracker): Tracker for logging metrics.
            n_splits (int): Number of cross-validation splits.
            train_size (int, optional): Maximum size of the training set. Defaults to False.
        """
        self.model = model
        self.tracker = tracker
        self.n_splits = n_splits
        self.train_size = train_size
        self.models = []

    def fit(
        self,
        df: pl.DataFrame,
        verbose: bool = False,
    ) -> list:
        """Fits models on cross-validation folds.

        Args:
            df (pl.DataFrame): DataFrame containing the data.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            list: Scores for each fold.
        """
        dates_unique = df.select(pl.col(COL_DATE).unique().sort()).to_series().to_numpy()

        test_size = (
            TEST_SIZE
            if len(dates_unique) > TEST_SIZE * (self.n_splits + 1)
            else len(dates_unique) // (self.n_splits + 1)
        ) # For testing purposes on small samples
        cv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=test_size,
            max_train_size=self.train_size
        )
        cv_split = cv.split(dates_unique)

        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv_split):
            if verbose:
                print("-"*20 + f"Fold {fold}" + "-"*20)
                print(
                    f"Train dates from {dates_unique[train_idx].min()}"
                    f" to {dates_unique[train_idx].max()}"
                )
                print(
                    f"Valid dates from {dates_unique[valid_idx].min()}"
                    f" to {dates_unique[valid_idx].max()}"
                )

            dates_train = dates_unique[train_idx]
            dates_valid = dates_unique[valid_idx]

            df_train = df.filter(pl.col(COL_DATE).is_in(dates_train))
            df_valid = df.filter(pl.col(COL_DATE).is_in(dates_valid))

            model_fold = clone(self.model)
            model_fold.set_run_name(f"fold{fold}")
            model_fold.fit(df_train, df_valid, verbose=verbose)

            self.models.append(model_fold)

            preds = []
            cnt_dates = 0
            model_save = copy.deepcopy(model_fold)
            for date_id in tqdm(dates_valid):
                df_valid_date = df_valid.filter(pl.col(COL_DATE) == date_id)

                if model_fold.refit & (cnt_dates > 0):
                    df_upd = df.filter(pl.col(COL_DATE)==date_id-1)
                    if len(df_upd) > 0:
                        model_save.update(df_upd)

                preds_i, _ = model_save.predict(df_valid_date)
                preds += list(preds_i)
                cnt_dates += 1
            preds = np.array(preds)

            df_valid = df_valid.fill_null(0.0)
            y_true = df_valid.select(pl.col(model_fold.col_target)).to_series().to_numpy()
            weights = df_valid.select(pl.col(COL_WEIGHT)).to_series().to_numpy()
            score = r2_weighted(y_true, preds, weights)
            scores.append(score)

            print(f"R2: {score:.5f}")
            if self.tracker:
                self.tracker.log_metrics({f"fold_{fold}": score})

        if self.tracker:
            self.tracker.log_metrics({"cv": np.mean(scores)})
        return scores

    def load(self) -> None:
        """Loads models for each fold from disk.
        """
        self.models = []
        for i in range(self.n_splits):
            model = clone(self.model)
            model.set_run_name(f"fold{i}")
            model.fit()
            self.models.append(model)
