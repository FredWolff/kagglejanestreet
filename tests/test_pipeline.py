"""Tests for janestreet.pipeline."""

import weakref

import numpy as np
import polars as pl

from janestreet import pipeline
from janestreet.models.nn import NN
from janestreet.pipeline import FullPipeline, _feature_matrix
from janestreet.transformers import PolarsTransformer

N_SYMBOLS = 2
T = 968  # time_ids per date, fixed by CustomTensorDataset
FEATURES = ["feature_00", "feature_01", "feature_time_id"]
RESPONDERS = [f"responder_{i}" for i in range(11)]


def _scaled_frame():
    df = pl.DataFrame({
        "feature_00": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
        "feature_01": pl.Series([0.5, None, 1.5, 9.0], dtype=pl.Float32),
        "feature_time_id": pl.Series([0, 1, 2, 3], dtype=pl.Int16),
    })
    features = ["feature_00", "feature_01", "feature_time_id"]
    return PolarsTransformer(features).fit_transform(df), features


def test_feature_matrix_stays_float32_where_plain_to_numpy_upcasts_to_float64():
    # The scaled Int16 feature_time_id becomes Float64; mixed with Float32 columns,
    # to_numpy() then returns a float64 matrix twice the size.
    df, features = _scaled_frame()
    assert df.select(features).to_numpy().dtype == np.float64

    assert _feature_matrix(df, features).dtype == np.float32


def test_feature_matrix_values_match_what_the_network_saw_before():
    # NN converts its input to float32, so the float32 matrix must equal the old float64
    # matrix cast to float32, bit for bit.
    df, features = _scaled_frame()

    old = df.select(features).to_numpy().astype(np.float32)

    np.testing.assert_array_equal(_feature_matrix(df, features), old)


def _frame(dates):
    """Complete (date, time, symbol) grid in the same row order as the real train data."""
    rng = np.random.default_rng(0)
    n = len(dates) * T * N_SYMBOLS
    time_id = np.tile(np.repeat(np.arange(T), N_SYMBOLS), len(dates)).astype(np.int16)
    data = {
        "symbol_id": np.tile(np.arange(N_SYMBOLS), len(dates) * T).astype(np.int8),
        "date_id": np.repeat(dates, T * N_SYMBOLS).astype(np.int16),
        "time_id": time_id,
        "weight": rng.random(n, dtype=np.float32) + 0.5,
        "feature_00": rng.standard_normal(n, dtype=np.float32),
        "feature_01": rng.standard_normal(n, dtype=np.float32),
        "feature_time_id": time_id.copy(),
    }
    for r in RESPONDERS:
        data[r] = rng.standard_normal(n, dtype=np.float32)
    return pl.DataFrame(data)


def _small_nn():
    return NN(
        model_type="gru", hidden_sizes=[4], dropout_rates=[0.0, 0.0, 0.0],
        hidden_sizes_linear=[4, 3], dropout_rates_linear=[0.0, 0.0], epochs=1,
    )


def test_feature_matrices_are_freed_before_training_starts(monkeypatch):
    # X_train / X_valid are the largest objects of a fold. They must not stay referenced while
    # the network trains - the datasets hold their own float32 copies.
    refs = []
    real_feature_matrix = pipeline._feature_matrix

    def recording_feature_matrix(df, features):
        X = real_feature_matrix(df, features)
        refs.append(weakref.ref(X))
        return X

    monkeypatch.setattr(pipeline, "_feature_matrix", recording_feature_matrix)

    alive_during_training = []

    class ProbeNN(NN):
        def train_one_epoch(self, *args, **kwargs):
            # No gc.collect() here on purpose: refcounting alone must already have freed them.
            alive_during_training.extend(ref() is not None for ref in refs)
            return super().train_one_epoch(*args, **kwargs)

    model = ProbeNN(
        model_type="gru", hidden_sizes=[4], dropout_rates=[0.0, 0.0, 0.0],
        hidden_sizes_linear=[4, 3], dropout_rates_linear=[0.0, 0.0], epochs=1,
    )
    fp = FullPipeline(
        model, preprocessor=PolarsTransformer(FEATURES), run_name="t", name="t",
        features=FEATURES, save_to_disc=False,
    )

    fp.fit(_frame([0, 1, 2]), _frame([3, 4]))

    assert len(refs) == 2  # X_train and X_valid
    assert alive_during_training == [False, False]


def test_nn_fit_empties_list_inputs_but_leaves_tuples_alone():
    def arrays(dates):
        df = _frame(dates)
        responders = [r for r in RESPONDERS if r != "responder_6"]
        return [
            df.select(FEATURES).cast(pl.Float32).to_numpy(), df.select(responders).to_numpy(),
            df["responder_6"].to_numpy(), df["weight"].to_numpy(), df["symbol_id"].to_numpy(),
            df["date_id"].to_numpy(), df["time_id"].to_numpy(),
        ]

    train, valid = arrays([0, 1]), arrays([2])
    _small_nn().fit(train, valid)
    assert train == [] and valid == []

    train_t, valid_t = tuple(arrays([0, 1])), tuple(arrays([2]))
    _small_nn().fit(train_t, valid_t)
    assert len(train_t) == 7 and len(valid_t) == 7
