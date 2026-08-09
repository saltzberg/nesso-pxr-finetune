import numpy as np
import pandas as pd

from nesso_pxr.comparison import (
    clustered_bootstrap,
    error_complementarity,
    inverse_se_weights,
    regression_metrics,
)


def test_regression_metrics_and_calibration() -> None:
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = observed.copy()
    metrics = regression_metrics(observed, predicted)
    assert metrics["mae"] == 0.0
    assert metrics["rae"] == 0.0
    assert np.isclose(metrics["spearman"], 1.0)
    assert np.isclose(metrics["calibration_intercept"], 0.0)
    assert np.isclose(metrics["calibration_slope"], 1.0)


def test_inverse_se_weights_match_established_recipe() -> None:
    weights = inverse_se_weights([1.0, 0.5, 0.25], [1.0, 0.5, 2.0])
    assert np.allclose(weights, [1.0, 1.0, 8.0])


def test_cluster_bootstrap_is_reproducible_and_paired() -> None:
    frame = pd.DataFrame(
        {
            "pEC50": [1.0, 2.0, 3.0, 4.0],
            "first": [1.1, 2.1, 2.9, 3.9],
            "second": [1.2, 1.8, 3.2, 3.8],
            "group": ["a", "a", "b", "c"],
        }
    )
    first, draws_first = clustered_bootstrap(
        frame,
        {"first": "first", "second": "second"},
        group_column="group",
        replicates=20,
        seed=7,
    )
    second, draws_second = clustered_bootstrap(
        frame,
        {"first": "first", "second": "second"},
        group_column="group",
        replicates=20,
        seed=7,
    )
    pd.testing.assert_frame_equal(first, second)
    assert np.array_equal(draws_first["first:mae"], draws_second["first:mae"])


def test_error_complementarity_identical_models() -> None:
    frame = pd.DataFrame(
        {
            "pEC50": [1.0, 2.0, 3.0, 4.0],
            "first": [1.2, 1.8, 3.1, 3.5],
            "second": [1.2, 1.8, 3.1, 3.5],
        }
    )
    metrics = error_complementarity(frame, "first", "second")
    assert np.isclose(metrics["signed_residual_pearson"], 1.0)
    assert np.isclose(metrics["absolute_error_pearson"], 1.0)
    assert metrics["exact_tie_fraction"] == 1.0
