import numpy as np
import pandas as pd
import pytest

from nesso_pxr.transfer_ablation import (
    apply_linear_stack,
    crossfit_scalar_stacks,
    fit_convex_mae_weight,
    fit_linear_stack,
    huber_values,
    nested_ridge_probe,
    similarity_quantiles,
)


def test_huber_values_use_quadratic_and_linear_regions() -> None:
    residual = np.array([0.0, 0.25, 0.5, 1.0])
    observed = huber_values(residual, 0.5)
    assert observed.tolist() == pytest.approx([0.0, 0.03125, 0.125, 0.375])


def test_nested_ridge_probe_keeps_outer_groups_held_out() -> None:
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(5), 12)
    x = rng.normal(size=(len(groups), 4))
    y = 1.5 + x @ np.array([0.7, -0.2, 0.0, 0.4])
    result = nested_ridge_probe(x, y, groups, [0.0, 0.1, 1.0])
    assert len(result.predictions) == len(y)
    assert result.predictions["predicted"].notna().all()
    assert set(result.selections["outer_fold"]) == set(range(5))
    assert np.mean(np.abs(result.predictions["predicted"] - y)) < 1e-10


def test_convex_and_linear_stacks_recover_informative_first_model() -> None:
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    first = observed.copy()
    second = observed[::-1]
    weight = fit_convex_mae_weight(observed, first, second)
    assert weight > 0.99
    coefficients = fit_linear_stack(observed, first, second)
    predicted = apply_linear_stack(coefficients, first, second)
    assert predicted == pytest.approx(observed)


def test_crossfit_scalar_stacks_produce_every_prediction() -> None:
    frame = pd.DataFrame(
        {
            "pEC50": np.arange(15, dtype=float),
            "first": np.arange(15, dtype=float),
            "second": np.zeros(15),
            "fold": np.repeat(np.arange(5), 3),
        }
    )
    predictions, coefficients = crossfit_scalar_stacks(
        frame,
        observed_column="pEC50",
        first_column="first",
        second_column="second",
        group_column="fold",
    )
    assert predictions[["convex_stack", "linear_stack"]].notna().all().all()
    assert len(coefficients) == 5


def test_similarity_quantiles_include_threshold_fractions() -> None:
    frame = pd.DataFrame(
        {
            "fold": [0, 0, 1, 1],
            "similarity": [0.2, 0.6, 0.7, 0.9],
        }
    )
    result = similarity_quantiles(
        frame,
        similarity_column="similarity",
        group_columns=["fold"],
    )
    fold_zero = result.loc[result["fold"].eq(0)].iloc[0]
    assert fold_zero["minimum"] == pytest.approx(0.2)
    assert fold_zero["maximum"] == pytest.approx(0.6)
    assert fold_zero["fraction_ge_0_6"] == pytest.approx(0.5)
