"""Tests for janestreet.wandb_analysis."""

from pathlib import Path

import pytest

from janestreet.wandb_analysis import WandbRuns


class FakeRun:
    def __init__(self, id, name, state, created_at, config, summary):
        self.id = id
        self.name = name
        self.state = state
        self.created_at = created_at
        self.config = config
        self.summary = summary


class FakeArtifact:
    def __init__(self, type, files):
        self.type = type
        self._files = files

    def download(self, root):
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        for name, content in self._files.items():
            (root_path / name).write_text(content)
        return str(root_path)


class FakeRunDetail:
    def __init__(self, artifacts):
        self._artifacts = artifacts

    def logged_artifacts(self):
        return self._artifacts


class FakeApi:
    def __init__(self, runs, default_entity="frederikwolff", run_detail=None):
        self._runs = runs
        self.default_entity = default_entity
        self.calls = []
        self.run_calls = []
        self._run_detail = run_detail

    def runs(self, path, filters=None, lazy=True):
        self.calls.append({"path": path, "filters": filters, "lazy": lazy})
        return self._runs

    def run(self, path):
        self.run_calls.append(path)
        return self._run_detail


def test_leaderboard_returns_one_row_per_finished_run():
    run = FakeRun(
        id="abc123",
        name="gru_1.0_700_cv",
        state="finished",
        created_at="2026-01-01T00:00:00",
        config={"category": "model_ver27_ts", "comment": "", "lr": 0.0005},
        summary={"fold_0": 0.01, "fold_1": 0.02, "cv": 0.015},
    )
    api = FakeApi(runs=[run])

    wandb_runs = WandbRuns(api=api, project="kaggle_janestreet")
    result = wandb_runs.leaderboard()

    assert result["run_id"].to_list() == ["abc123"]
    assert result["name"].to_list() == ["gru_1.0_700_cv"]
    assert result["category"].to_list() == ["model_ver27_ts"]
    assert result["cv"].to_list() == [0.015]


def test_leaderboard_sorts_multiple_runs_by_cv_descending():
    runs = [
        FakeRun("a", "run_a", "finished", "t0", {"category": "c1"}, {"cv": 0.01}),
        FakeRun("b", "run_b", "finished", "t0", {"category": "c1"}, {"cv": 0.05}),
        FakeRun("c", "run_c", "finished", "t0", {"category": "c1"}, {"cv": 0.03}),
    ]
    api = FakeApi(runs=runs)

    result = WandbRuns(api=api, project="kaggle_janestreet").leaderboard()

    assert result["run_id"].to_list() == ["b", "c", "a"]


def test_pull_queries_default_entity_project_and_finished_state_filter():
    api = FakeApi(runs=[])

    WandbRuns(api=api, project="kaggle_janestreet").pull()

    assert api.calls == [
        {
            "path": "frederikwolff/kaggle_janestreet",
            "filters": {"state": "finished"},
            "lazy": False,
        }
    ]


def test_pull_adds_category_filter_when_given():
    api = FakeApi(runs=[])

    WandbRuns(api=api, project="kaggle_janestreet", category="model_ver27_ts").pull()

    assert api.calls[0]["filters"] == {
        "state": "finished",
        "config.category": "model_ver27_ts",
    }


def test_leaderboard_filters_by_category_and_comment_after_pull():
    runs = [
        FakeRun("a", "run_a", "finished", "t0", {"category": "c1", "comment": "x"}, {"cv": 0.05}),
        FakeRun("b", "run_b", "finished", "t0", {"category": "c2", "comment": "x"}, {"cv": 0.09}),
        FakeRun("c", "run_c", "finished", "t0", {"category": "c1", "comment": "y"}, {"cv": 0.07}),
    ]
    api = FakeApi(runs=runs)

    result = WandbRuns(api=api, project="kaggle_janestreet").leaderboard(category="c1", comment="x")

    assert result["run_id"].to_list() == ["a"]


def test_pull_flattens_list_valued_config_fields_into_indexed_columns():
    run = FakeRun(
        "a", "run_a", "finished", "t0",
        config={"category": "c1", "hidden_sizes": [500, 300]},
        summary={"cv": 0.05},
    )
    api = FakeApi(runs=[run])

    result = WandbRuns(api=api, project="kaggle_janestreet").pull()

    assert result["hidden_sizes_0"].to_list() == [500]
    assert result["hidden_sizes_1"].to_list() == [300]
    assert "hidden_sizes" not in result.columns


def test_correlation_ranks_numeric_config_fields_excluding_strings_and_constants():
    runs = [
        FakeRun("a", "run_a", "finished", "t0", {"category": "c1", "lr": 0.1, "epochs": 10}, {"cv": 0.1}),
        FakeRun("b", "run_b", "finished", "t0", {"category": "c1", "lr": 0.2, "epochs": 10}, {"cv": 0.2}),
        FakeRun("c", "run_c", "finished", "t0", {"category": "c1", "lr": 0.3, "epochs": 10}, {"cv": 0.3}),
    ]
    api = FakeApi(runs=runs)

    result = WandbRuns(api=api, project="kaggle_janestreet").correlation()

    assert result["column"].to_list() == ["lr"]
    assert result["correlation"].to_list() == pytest.approx([1.0])


def test_fold_stability_ranks_runs_by_combined_mean_and_std_rank():
    runs = [
        FakeRun("a", "run_a", "finished", "t0", {}, {"fold_0": 0.4, "fold_1": 0.5, "fold_2": 0.6}),
        FakeRun("b", "run_b", "finished", "t0", {}, {"fold_0": 0.28, "fold_1": 0.3, "fold_2": 0.32}),
        FakeRun("c", "run_c", "finished", "t0", {}, {"fold_0": 0.05, "fold_1": 0.1, "fold_2": 0.15}),
    ]
    api = FakeApi(runs=runs)

    result = WandbRuns(api=api, project="kaggle_janestreet").fold_stability()

    # a: mean 0.5, std 0.1 (best mean, worst stability)
    # b: mean 0.3, std 0.02 (mid mean, best stability) -> wins on combined rank
    # c: mean 0.1, std 0.05 (worst mean, mid stability)
    assert result["run_id"].to_list() == ["b", "a", "c"]
    assert result["mean_cv"].to_list() == pytest.approx([0.3, 0.5, 0.1])
    assert result["std_cv"].to_list() == pytest.approx([0.02, 0.1, 0.05])


def test_category_comparison_reports_best_and_median_cv_per_category():
    runs = [
        FakeRun("a", "run_a", "finished", "t0", {"category": "c1"}, {"cv": 0.1}),
        FakeRun("b", "run_b", "finished", "t0", {"category": "c1"}, {"cv": 0.3}),
        FakeRun("c", "run_c", "finished", "t0", {"category": "c2"}, {"cv": 0.2}),
    ]
    api = FakeApi(runs=runs)

    result = WandbRuns(api=api, project="kaggle_janestreet").category_comparison()

    assert result.sort("category")["category"].to_list() == ["c1", "c2"]
    assert result.sort("category")["best_cv"].to_list() == pytest.approx([0.3, 0.2])
    assert result.sort("category")["median_cv"].to_list() == pytest.approx([0.2, 0.2])


def test_analyze_returns_all_four_views_together():
    run = FakeRun(
        "a", "run_a", "finished", "t0",
        config={"category": "c1", "lr": 0.1},
        summary={"fold_0": 0.1, "fold_1": 0.3, "cv": 0.2},
    )
    api = FakeApi(runs=[run])

    result = WandbRuns(api=api, project="kaggle_janestreet").analyze()

    assert set(result.keys()) == {
        "leaderboard", "correlation", "fold_stability", "category_comparison",
    }
    assert result["leaderboard"]["run_id"].to_list() == ["a"]
    assert result["fold_stability"]["run_id"].to_list() == ["a"]
    assert result["category_comparison"]["category"].to_list() == ["c1"]
    assert api.run_calls == []


def test_get_run_features_downloads_the_feature_list_artifact_for_one_run(tmp_path):
    artifact = FakeArtifact(type="dataset", files={"features.txt": "feature_00\nfeature_01\n"})
    run_detail = FakeRunDetail(artifacts=[artifact])
    api = FakeApi(runs=[], run_detail=run_detail)

    wandb_runs = WandbRuns(api=api, project="kaggle_janestreet")
    features = wandb_runs.get_run_features("abc123", tmp_dir=tmp_path)

    assert features == ["feature_00", "feature_01"]
    assert api.run_calls == ["frederikwolff/kaggle_janestreet/abc123"]


def test_get_run_features_is_never_triggered_by_bulk_pull_or_analyze():
    run = FakeRun(
        "a", "run_a", "finished", "t0",
        config={"category": "c1"},
        summary={"fold_0": 0.1, "cv": 0.1},
    )
    api = FakeApi(runs=[run])

    wandb_runs = WandbRuns(api=api, project="kaggle_janestreet")
    wandb_runs.pull()
    wandb_runs.analyze()

    assert api.run_calls == []


def test_analyze_handles_zero_pulled_runs_without_crashing():
    api = FakeApi(runs=[])

    result = WandbRuns(api=api, project="kaggle_janestreet").analyze()

    assert result["leaderboard"].height == 0
    assert result["correlation"].height == 0
    assert result["fold_stability"].height == 0
    assert result["category_comparison"].height == 0


def test_fold_stability_handles_runs_with_no_fold_metrics_without_crashing():
    run = FakeRun("a", "run_a", "finished", "t0", {"category": "c1"}, {"cv": 0.1})
    api = FakeApi(runs=[run])

    result = WandbRuns(api=api, project="kaggle_janestreet").fold_stability()

    assert result.height == 0


def test_analyze_pulls_the_run_list_exactly_once():
    run = FakeRun(
        "a", "run_a", "finished", "t0",
        config={"category": "c1"},
        summary={"fold_0": 0.1, "fold_1": 0.2, "cv": 0.15},
    )
    api = FakeApi(runs=[run])

    WandbRuns(api=api, project="kaggle_janestreet").analyze()

    assert len(api.calls) == 1


def test_pull_never_lets_a_summary_metric_overwrite_an_identity_column():
    run = FakeRun(
        "a", "run_a", "finished", "t0",
        config={"category": "c1"},
        # a run logging a summary metric literally called "name" must not
        # clobber the run's actual name.
        summary={"cv": 0.1, "name": "not-the-real-name"},
    )
    api = FakeApi(runs=[run])

    result = WandbRuns(api=api, project="kaggle_janestreet").pull()

    assert result["name"].to_list() == ["run_a"]


def test_correlation_excludes_numeric_summary_metrics_that_are_not_config_fields():
    runs = [
        FakeRun("a", "run_a", "finished", "t0", {"lr": 0.1}, {"cv": 0.1, "best_iteration": 5}),
        FakeRun("b", "run_b", "finished", "t0", {"lr": 0.2}, {"cv": 0.2, "best_iteration": 50}),
        FakeRun("c", "run_c", "finished", "t0", {"lr": 0.3}, {"cv": 0.3, "best_iteration": 500}),
    ]
    api = FakeApi(runs=runs)

    result = WandbRuns(api=api, project="kaggle_janestreet").correlation()

    # best_iteration is a training-time result, not a hyperparameter, and is
    # even more perfectly correlated with cv than lr here -- it must not
    # appear, or it would rank above lr and mislead as an actionable setting.
    assert result["column"].to_list() == ["lr"]
