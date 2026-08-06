import pytest

from nesso_pxr.smoke import allocate_fold_counts, evenly_spaced_indices


def test_allocate_fold_counts_balances_sixteen_across_five_folds() -> None:
    assert allocate_fold_counts(16, [0, 1, 2, 3, 4]) == {
        0: 4,
        1: 3,
        2: 3,
        3: 3,
        4: 3,
    }


def test_evenly_spaced_indices_cover_range_deterministically() -> None:
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced_indices(9, 1) == [4]


def test_evenly_spaced_indices_reject_invalid_request() -> None:
    with pytest.raises(ValueError):
        evenly_spaced_indices(3, 4)
