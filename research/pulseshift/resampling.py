import numpy as np
import pandas as pd


def valid_draw_count(count) -> bool:
    return (
        not isinstance(count, (bool, np.bool_))
        and isinstance(count, (int, np.integer))
        and bool(count >= 2)
    )


def cluster_members(groups, row_count: int) -> list[np.ndarray] | None:
    if groups is None:
        return None
    groups = np.asarray(groups, dtype=object)
    if groups.ndim != 1 or len(groups) != row_count:
        raise ValueError("bootstrap groups must align with rows")
    if pd.isna(groups).any():
        raise ValueError("bootstrap groups must not contain missing values")
    codes, names = pd.factorize(groups, sort=True)
    return [np.flatnonzero(codes == code) for code in range(len(names))]


def row_samples(row_count: int, count: int, rng, members=None):
    for _ in range(count):
        if members is None:
            yield rng.choice(row_count, size=row_count, replace=True)
        else:
            selected = rng.integers(0, len(members), len(members))
            yield np.concatenate([members[group] for group in selected])


def sufficient_draws(valid: int, requested: int) -> bool:
    return valid >= max(2, int(np.ceil(0.8 * requested)))
