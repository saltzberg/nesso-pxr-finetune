import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nesso_pxr.binary_head_reanalysis import (
    assert_outer_predictions_isolated,
    family_bootstrap_proxy_metrics,
    functional_potency_proxy,
    functional_proxy_metrics,
    safe_logit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_proxy_thresholds_are_prespecified_and_inclusive() -> None:
    observed = np.array([3.99, 4.0, 5.0, 6.0])
    assert functional_potency_proxy(observed, 4.0).tolist() == [
        False,
        True,
        True,
        True,
    ]
    with pytest.raises(ValueError, match="not prespecified"):
        functional_potency_proxy(observed, 4.5)


def test_proxy_contingency_counts_do_not_invent_binding_truth() -> None:
    observed = np.array([6.1, 5.1, 4.9, 3.0])
    probability = np.array([0.8, 0.2, 0.7, 0.1])
    metrics = functional_proxy_metrics(observed, probability, 5.0)
    assert metrics["proxy_positive_binder_positive"] == 1
    assert metrics["proxy_positive_binder_negative"] == 1
    assert metrics["proxy_negative_binder_positive"] == 1
    assert metrics["proxy_negative_binder_negative"] == 1
    assert not any("true_" in key or "false_" in key for key in metrics)


def test_ensemble_logit_matches_logit_of_mean_member_probability() -> None:
    member_logits = np.array([[-2.0, 1.0], [0.5, -0.5]])
    member_probabilities = 1.0 / (1.0 + np.exp(-member_logits))
    ensemble_probability = member_probabilities.mean(axis=1)
    ensemble_logit = safe_logit(ensemble_probability)
    recovered = 1.0 / (1.0 + np.exp(-ensemble_logit))
    assert recovered == pytest.approx(ensemble_probability)


def test_family_bootstrap_proxy_metrics_reports_cluster_count() -> None:
    frame = pd.DataFrame(
        {
            "family": np.repeat(np.arange(6), 2),
            "pEC50": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5] * 2,
            "binder_probability": np.linspace(0.05, 0.95, 12),
        }
    )
    intervals = family_bootstrap_proxy_metrics(
        frame,
        group_column="family",
        thresholds=(4.0,),
        replicates=20,
        seed=7,
    )
    assert set(intervals["metric"]) == {"roc_auc", "average_precision"}
    assert set(intervals["bootstrap_clusters"]) == {6}
    assert set(intervals["bootstrap_replicates"]) == {20}


def test_outer_prediction_isolation_rejects_duplicates_and_missing_values() -> None:
    valid = pd.DataFrame(
        {
            "feature_row": [1, 2, 3],
            "fold": [0, 1, 2],
            "probe_predicted_pEC50": [4.0, 5.0, 6.0],
        }
    )
    assert_outer_predictions_isolated(valid, expected_rows=[1, 2, 3])
    with pytest.raises(AssertionError, match="multiple"):
        assert_outer_predictions_isolated(pd.concat([valid, valid.iloc[[0]]]))
    invalid = valid.copy()
    invalid.loc[1, "probe_predicted_pEC50"] = np.nan
    with pytest.raises(AssertionError, match="incomplete"):
        assert_outer_predictions_isolated(invalid)


def test_generated_reconstruction_provenance_has_exact_continuous_parity() -> None:
    path = (
        REPO_ROOT
        / "artifacts"
        / "experiments"
        / "binary_head_reanalysis"
        / "reconstruction_provenance.json"
    )
    if not path.is_file():
        pytest.skip("binary reconstruction artifacts have not been generated")
    provenance = json.loads(path.read_text())
    assert provenance["status"] == "pass"
    assert provenance["coverage"]["labeled_total"] == 4134
    assert provenance["coverage"]["challenge_total"] == 513
    continuous = provenance["continuous_replay"]
    labeled_errors = continuous["labeled"].values()
    challenge_errors = continuous["challenge"].values()
    pair_errors = [
        error
        for variant in continuous["pair_pool"].values()
        for error in variant.values()
    ]
    assert max([*labeled_errors, *challenge_errors, *pair_errors]) <= 1e-6
