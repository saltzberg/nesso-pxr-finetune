"""Reporting-only safeguards; fixtures test mathematics, not model performance."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/20260906_low_data_followup/code/analyze_final.py"
)
spec = importlib.util.spec_from_file_location("final_analysis", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_roundtrip_and_literal_identity(tmp_path):
    frame = pd.DataFrame(
        {"record_id": ["NA", "001"], "value": [1.0000000000000002, np.inf]}
    )
    m.save_csv(tmp_path / "x.csv", frame)
    got = m.read_csv(tmp_path / "x.csv")
    assert got.record_id.tolist() == ["NA", "001"]
    np.testing.assert_array_equal(got.value, frame.value)


def test_compression_direction_and_bias():
    r = m.regression(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.5, 3.0]))
    assert r["pred_on_observed_slope"] == 0.5
    assert r["observed_on_pred_slope"] == 2
    assert r["sd_ratio"] == 0.5


def test_infinity_is_covered_not_missing():
    f = pd.DataFrame(
        {
            "y_true": [1.0, 4.0],
            "y_pred": [2.0, 2.0],
            "lower_90": [-np.inf] * 2,
            "upper_90": [np.inf] * 2,
        }
    )
    r = m.interval_stats(
        f, {"width_90": None, "interval_90_finite": False, "coverage_90": 1.0}, 90
    )
    assert r["width_90"] == np.inf and r["unbounded_rows_90"] == 2
    assert r["coverage_90"] == 1.0


def test_repeats_do_not_inflate_effective_sample_size():
    f = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "chemical_group": [1, 2],
            "y_true": [1.0, 2.0],
            "y_pred": [2.0, 3.0],
            "weight": [1.0, 2.0],
        }
    )
    one = m.summarize_predictions(f)
    ten = m.summarize_predictions(pd.concat([f] * 10))
    assert one["effective_n_identity"] == ten["effective_n_identity"]
    assert ten["unique_compounds"] == 2 and ten["prediction_rows"] == 20
    assert ten["sparse"]


def test_empty_stratum_retained():
    assert m.summarize_predictions(pd.DataFrame())["prediction_rows"] == 0
    assert m.summarize_predictions(pd.DataFrame())["sparse"]


def test_paired_conditional_zero_and_fixed_draws():
    a = pd.DataFrame(
        {
            "record_id": [str(i) for i in range(100)],
            "outer_fold": np.arange(100) % 5,
            "chemical_group": np.arange(100),
            "weight": np.ones(100),
            "draw_count": np.full(100, 10),
            "abs_error": np.linspace(0, 1, 100),
        }
    )
    r = m.conditional_bootstrap(a, a, repeats=200)
    assert (
        r["conditional_delta"] == r["conditional_low95"] == r["conditional_high95"] == 0
    )
    assert r["unique_compounds"] == r["resampling_groups"] == 100
    bad = a.copy()
    bad.loc[0, "draw_count"] = 9
    with pytest.raises(AssertionError):
        m.conditional_bootstrap(a, bad, repeats=5)


def test_constant_prediction_slope_undefined():
    r = m.regression([1, 2, 3], [2, 2, 2])
    assert np.isnan(r["observed_on_pred_slope"])
    assert r["sd_ratio"] == 0
