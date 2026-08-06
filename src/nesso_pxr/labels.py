"""Numerical label contracts for Nesso and PXR endpoints."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

ArrayLike = TypeVar("ArrayLike")


def pec50_to_nesso_scale(value: ArrayLike) -> ArrayLike:
    """Convert pEC50 to Nesso's documented log10(micromolar) scale.

    This is a numerical sign/offset alignment only. It does not assert that a
    functional PXR EC50 is biologically equivalent to Nesso's IC50-like target.
    """

    return 6.0 - value  # type: ignore[operator, return-value]


def nesso_scale_to_pec50(value: ArrayLike) -> ArrayLike:
    """Convert Nesso's documented output scale to pEC50 units."""

    return 6.0 - value  # type: ignore[operator, return-value]


def effective_uncertainty_weight(
    assay_standard_error: np.ndarray,
    *,
    floor: float,
    cap_quantile: float = 0.95,
    curation_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Return mean-one inverse-variance weights with a fitted noise floor.

    The caller must fit/select ``floor`` and the cap using training data only.
    ``cap_quantile`` is applied before optional curation weights, matching the
    prespecified assay-uncertainty contract.
    """

    sigma = np.asarray(assay_standard_error, dtype=np.float64)
    if sigma.ndim != 1 or sigma.size == 0:
        raise ValueError("assay_standard_error must be a non-empty 1D array")
    if not np.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError("assay_standard_error must be finite and nonnegative")
    if floor <= 0:
        raise ValueError("floor must be positive")
    if not 0 < cap_quantile <= 1:
        raise ValueError("cap_quantile must be in (0, 1]")

    raw = 1.0 / (np.square(sigma) + floor**2)
    cap = np.quantile(raw, cap_quantile)
    weights = np.minimum(raw, cap)
    weights /= weights.mean()

    if curation_weight is not None:
        curation = np.asarray(curation_weight, dtype=np.float64)
        if curation.shape != weights.shape:
            raise ValueError("curation_weight must match assay_standard_error")
        if not np.isfinite(curation).all() or (curation < 0).any():
            raise ValueError("curation_weight must be finite and nonnegative")
        weights *= curation
        if weights.mean() == 0:
            raise ValueError("combined weights cannot all be zero")
        weights /= weights.mean()

    return weights
