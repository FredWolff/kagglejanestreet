"""Tests for janestreet.tracker."""

from janestreet.tracker import WandbTracker
from janestreet.config import WANDB_PROJECT


class FakeRunDetail:
    def __init__(self):
        self.summary = {}
        self.settings = {}

    def update(self):
        pass


class FakeApi:
    def __init__(self, default_entity):
        self.default_entity = default_entity
        self.run_calls = []
        self._run_detail = FakeRunDetail()

    def run(self, path):
        self.run_calls.append(path)
        return self._run_detail


def test_update_summary_resolves_entity_from_api_default_entity():
    api = FakeApi(default_entity="frederikwolff")
    tracker = WandbTracker("run1", {}, category="c", comment="", api=api)

    tracker.update_summary("abc123", {"cv": 0.5})

    assert api.run_calls == [f"frederikwolff/{WANDB_PROJECT}/abc123"]


def test_update_settings_resolves_entity_from_api_default_entity():
    api = FakeApi(default_entity="frederikwolff")
    tracker = WandbTracker("run1", {}, category="c", comment="", api=api)

    tracker.update_settings("abc123", {"some_setting": True})

    assert api.run_calls == [f"frederikwolff/{WANDB_PROJECT}/abc123"]


def test_entity_override_takes_precedence_over_api_default_entity():
    api = FakeApi(default_entity="frederikwolff")
    tracker = WandbTracker("run1", {}, category="c", comment="", api=api, entity="shared-team")

    tracker.update_summary("abc123", {"cv": 0.5})
    tracker.update_settings("abc123", {"some_setting": True})

    assert api.run_calls == [
        f"shared-team/{WANDB_PROJECT}/abc123",
        f"shared-team/{WANDB_PROJECT}/abc123",
    ]
