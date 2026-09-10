"""Regression tests for frozen-run reporting-only interval replay."""

import numpy as np
import pandas as pd
import pytest

from scripts.report_low_data_v2 import replay_interval


def frame_for(p, radius, y):
    return pd.DataFrame(
        {
            "record_id": ["fixture"],
            "y_pred": [p],
            "y_true": [y],
            "lower_90": [p - radius],
            "upper_90": [p + radius],
        }
    )


def test_boundary_replays_residual_not_rounded_bounds():
    p, y = 4.784778616304488, 1.82
    radius = abs(y - p)
    frame = frame_for(p, radius, y)
    # Fixture is an exact residual-boundary case, not a widened interval.
    audit = []
    result = replay_interval(
        frame, 90, {"width_90": 2 * radius, "task_id": "fixture"}, audit
    )
    assert result["coverage_90"] == 1.0
    assert result["width_90"] == 2 * radius
    assert len(audit) == 1
    assert audit[0]["serialized_bound_coverage"] == 0.0
    assert audit[0]["residual_coverage"] == 1.0


def test_endpoint_tampering_rejected():
    frame = frame_for(4.0, 1.0, 4.0)
    frame.loc[0, "lower_90"] = np.nextafter(3.0, -np.inf)
    with pytest.raises(ValueError, match="exactly replay"):
        replay_interval(frame, 90, {"width_90": 2.0}, [])


def test_not_widened_to_include_nearby_outside_point():
    frame = frame_for(0.0, 1.0, np.nextafter(1.0, np.inf))
    assert replay_interval(frame, 90, {"width_90": 2.0}, [])["coverage_90"] == 0.0


def test_unbounded_interval():
    frame = frame_for(4.0, np.inf, 5.0)
    result = replay_interval(
        frame, 90, {"width_90": None, "interval_90_finite": False}, []
    )
    assert result["coverage_90"] == 1.0
    assert result["width_90"] == np.inf
