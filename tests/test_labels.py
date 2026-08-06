import numpy as np
import pytest

from nesso_pxr.labels import (
    effective_uncertainty_weight,
    nesso_scale_to_pec50,
    pec50_to_nesso_scale,
)


def test_label_transform_round_trip() -> None:
    values = np.array([1.61, 4.65, 6.0, 7.5486])
    transformed = pec50_to_nesso_scale(values)
    np.testing.assert_allclose(nesso_scale_to_pec50(transformed), values)
    np.testing.assert_allclose(transformed, np.array([4.39, 1.35, 0.0, -1.5486]))


def test_uncertainty_weights_are_capped_and_normalized() -> None:
    sigma = np.array([0.00325, 0.1, 0.2, 0.7])
    weights = effective_uncertainty_weight(sigma, floor=0.15, cap_quantile=0.75)
    assert weights.mean() == pytest.approx(1.0)
    assert np.isfinite(weights).all()
    assert weights[0] == pytest.approx(weights.max())


def test_uncertainty_weights_apply_curation_and_renormalize() -> None:
    weights = effective_uncertainty_weight(
        np.array([0.1, 0.1]),
        floor=0.15,
        curation_weight=np.array([1.0, 0.5]),
    )
    np.testing.assert_allclose(weights, np.array([4 / 3, 2 / 3]))
