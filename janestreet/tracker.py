"""Evaluation and logging functions."""

import pandas as pd
import wandb

from .config import  WANDB_PROJECT, base_path

class WandbTracker:
    """Custom class for tracking experiments using WandB.

    This class provides methods for initializing runs, saving features and data, logging metrics,
    sending alerts, and updating run summaries and settings.

    Args:
        run_name (str): Name of the run.
        params (dict): Dictionary containing parameters for the run.
        category (str): Category of the run.
        comment (str): Comment or description for the run.
    """

    def __init__(
        self,
        run_name: str,
        params: dict,
        category: str,
        comment: str,
        api: object | None = None,
        entity: str | None = None,
    ) -> None:
        """Initializes the WandbTracker class.

        Args:
            run_name (str): Name of the run.
            params (dict): Dictionary containing parameters for the run.
            category (str): Category of the run.
            comment (str): Comment or description for the run.
            api (optional): Weights & Biases API client, used by
                `update_summary`/`update_settings` to look up other runs.
                Defaults to `wandb.Api()`. Injectable so tests can supply a
                fake client.
            entity (str, optional): W&B entity owning the run looked up by
                `update_summary`/`update_settings`. Defaults to the API
                client's default entity.
        """
        self.run_name = run_name
        self.params = params
        self.category = category
        self.comment = comment
        self.api = api if api is not None else wandb.Api()
        self.entity = entity or self.api.default_entity

    def init_run(self, features: list) -> None:
        """Initializes a new WandB run.

        Args:
            features (list): List of features used in the model.
        """
        config = self.params.copy()
        config.update({
            "model": "lgb",
            "category": self.category,
            "comment": self.comment,
            "n_features": len(features)
        })
        wandb.init(
            project=WANDB_PROJECT,
            name=self.run_name,
            config=config,
            dir=base_path,
            save_code=True
        )
        self.save_features(features)
        print(f"Running {self.run_name} model.")
        print(self.comment)

    def save_features(self, features: list) -> None:
        """Saves the list of features as a WandB artifact.

        Args:
            features (list): List of features used in the model.
        """
        feature_file_path = "features.txt"
        with open(feature_file_path, "w", encoding="utf-8") as file:
            for feature in features:
                file.write(f"{feature}\n")
        artifact = wandb.Artifact(
            name=f"{self.run_name}-feature-list",
            type="dataset"
        )
        artifact.add_file(feature_file_path)
        wandb.log_artifact(artifact)

    def save_data(self, df: pd.DataFrame, name: str) -> None:
        """Saves a DataFrame as a WandB artifact.

        Args:
            df (pd.DataFrame): DataFrame to be saved.
            name (str): Name of the artifact.
        """
        tab = wandb.Table(columns=list(df.columns), data=df.values.tolist())
        wandb.log({name: tab})

    def alert(self, text: str) -> None:
        """Sends an alert to the user via WandB.

        Args:
            text (str): Text of the alert.
        """
        wandb.alert(
            title=f'Run {self.run_name} finished.',
            text=text,
            level=wandb.AlertLevel.INFO
        )

    def log_metrics(self, metrics: dict) -> None:
        """Logs metrics to the current WandB run.

        Args:
            metrics (dict): Dictionary containing the metrics to log.
        """
        wandb.log(metrics)

    def update_summary(self, run_id: str, summary_params: dict) -> None:
        """Updates the summary of an existing WandB run.

        Args:
            run_id (str): ID of the WandB run.
            summary_params (dict): Dictionary containing summary parameters to update.
        """
        run = self.api.run(f"{self.entity}/{WANDB_PROJECT}/{run_id}")
        for key, val in summary_params.items():
            run.summary[key] = val
        run.summary.update()

    def update_settings(self, run_id: str, settings_params: dict) -> None:
        """Updates the settings of an existing WandB run.

        Args:
            run_id (str): ID of the WandB run.
            settings_params (dict): Dictionary containing settings parameters to update.
        """
        run = self.api.run(f"{self.entity}/{WANDB_PROJECT}/{run_id}")
        for key, val in settings_params.items():
            run.settings[key] = val
        run.update()

    def finish(self) -> None:
        """Finishes the current WandB run.
        """
        wandb.finish()
