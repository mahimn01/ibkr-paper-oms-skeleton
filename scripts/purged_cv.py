"""Purged + embargoed walk-forward cross-validation.

The exact thing v7/ATLAS lacked (it trained and validated on random overlapping
windows, fast_env._reset_env). For any model that predicts a forward-H-period
label, training rows whose label window overlaps the test block leak the answer;
serial correlation near the boundary leaks more. This yields expanding-window
walk-forward splits with:
  - PURGE   : drop train rows whose [i, i+label_horizon] label window reaches
              into the test block.
  - EMBARGO : drop an extra `embargo` rows immediately before the test block.
So the train/test gap is exactly (label_horizon + embargo) rows.

Indices are positions into a chronologically-sorted axis (e.g. rebalance dates).
Lopez de Prado, "Advances in Financial Machine Learning", Ch.7.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train: list[int]
    test: list[int]


def purged_walk_forward(
    n: int,
    *,
    n_splits: int = 5,
    label_horizon: int = 21,
    embargo: int = 10,
    min_train: int = 252,
    max_train: int | None = None,
) -> list[Split]:
    """Return purged+embargoed expanding (or rolling) walk-forward splits.

    n             : number of time steps (rows) on the sorted axis.
    n_splits      : number of contiguous OOS test blocks after the initial train.
    label_horizon : forward span of the label, in rows (purge width).
    embargo       : extra buffer rows dropped before each test block.
    min_train     : minimum train rows required to emit a split.
    max_train     : if set, use a rolling window of this many rows (else expanding).
    """
    if n_splits < 1 or n <= 0:
        return []
    gap = label_horizon + embargo
    test_region_start = min(min_train + gap, n)
    if test_region_start >= n:
        return []

    # Carve the remaining axis into n_splits contiguous test blocks.
    remaining = n - test_region_start
    block = max(1, remaining // n_splits)
    splits: list[Split] = []
    ts = test_region_start
    for k in range(n_splits):
        te = n if k == n_splits - 1 else min(ts + block, n)
        if ts >= te:
            break
        train_end = ts - gap  # purge + embargo gap before the test block
        if train_end < min_train:
            ts = te
            continue
        train_start = 0 if max_train is None else max(0, train_end - max_train)
        train = list(range(train_start, train_end))
        test = list(range(ts, te))
        if train and test:
            splits.append(Split(train=train, test=test))
        ts = te
    return splits


def _self_test() -> None:
    n, H, EM = 1000, 21, 10
    splits = purged_walk_forward(n, n_splits=5, label_horizon=H, embargo=EM, min_train=252)
    assert splits, "no splits produced"
    seen_test: set[int] = set()
    for s in splits:
        tr_end = s.train[-1]
        te_start = s.test[0]
        # Gap invariant: last train row + label window must not reach the test block,
        # plus the embargo buffer => gap >= H + EM.
        assert te_start - tr_end >= H + EM, f"gap too small: {te_start - tr_end} < {H+EM}"
        # No train row's label window overlaps the test block.
        assert tr_end + H < te_start, "label leakage into test"
        # Test blocks are disjoint and ordered.
        assert not (set(s.test) & seen_test), "overlapping test blocks"
        seen_test.update(s.test)
        # Train is strictly before test.
        assert max(s.train) < min(s.test)
    # Test blocks should tile the OOS region contiguously.
    assert min(seen_test) >= 252 + H + EM
    assert max(seen_test) == n - 1
    # Rolling-window variant respects max_train.
    roll = purged_walk_forward(n, n_splits=4, label_horizon=H, embargo=EM, min_train=252, max_train=300)
    for s in roll:
        assert len(s.train) <= 300
    print(f"OK: {len(splits)} expanding splits, gap>={H+EM}, no leakage; "
          f"train sizes {[len(s.train) for s in splits]}, test sizes {[len(s.test) for s in splits]}")


if __name__ == "__main__":
    _self_test()
