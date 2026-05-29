"""Regression tests for the architecture-aware ATLAS inference loader.

Covers:
- _is_v7_config discrimination (instance / dict / None)
- from_checkpoint round-trips for v1 (incl. legacy no-config) and v7
- the config/weights consistency guard
- end-to-end predict() on the real v7 checkpoint (skipped if absent)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.config_v7 import ATLASv7Config
from trading_algo.quant_core.models.atlas.model import ATLASModel
from trading_algo.quant_core.models.atlas.model_v7 import ATLASModelV7
from trading_algo.quant_core.models.atlas.inference import ATLASInference, _is_v7_config
from trading_algo.quant_core.models.atlas.execution_bridge import TradeDecision

V7_CKPT = "checkpoints/atlas_v7_ibkr_curriculum/atlas_v7_curriculum_final.pt"


def _synthetic_bars(n: int, seed: int = 0) -> list[tuple[datetime, float, float, float, float]]:
    # Trading bars only — weekdays (dow 0-4), matching CalendarEmbedding(5, ...).
    rng = np.random.default_rng(seed)
    price = 100.0
    day = datetime(2020, 1, 1)
    bars = []
    while len(bars) < n:
        if day.weekday() < 5:
            price *= 1 + rng.normal(0.0005, 0.012)
            high = price * (1 + abs(rng.normal(0, 0.004)))
            low = price * (1 - abs(rng.normal(0, 0.004)))
            vol = float(1_000_000 + rng.integers(0, 500_000))
            bars.append((day, price, high, low, vol))
        day += timedelta(days=1)
    return bars


def _feed_and_predict(atlas: ATLASInference, n: int, seed: int = 0) -> TradeDecision:
    decision: TradeDecision | None = None
    for d, p, h, lo, v in _synthetic_bars(n, seed):
        decision = atlas.predict(date=d, price=p, high=h, low=lo, volume=v)
    assert decision is not None
    return decision


def _assert_in_bounds(dec: TradeDecision) -> None:
    assert isinstance(dec, TradeDecision)
    assert -1.01 <= dec.direction <= 1.01
    assert dec.leverage >= -0.01
    assert -0.01 <= dec.delta <= 0.51
    assert dec.dte == 0 or 13.5 <= dec.dte <= 90.5
    assert -0.01 <= dec.profit_target <= 1.01


class TestIsV7Config:
    def test_instances(self) -> None:
        assert _is_v7_config(ATLASv7Config()) is True
        assert _is_v7_config(ATLASConfig()) is False

    def test_dicts(self) -> None:
        assert _is_v7_config({"patch_size": 5, "n_mlstm_layers": 2}) is True
        assert _is_v7_config({"n_mamba_layers": 4, "d_state": 16}) is False

    def test_none(self) -> None:
        assert _is_v7_config(None) is False


class TestSyntheticRoundTrip:
    def test_v1_no_config_defaults_to_v1(self, tmp_path) -> None:
        # Legacy v1 checkpoints stored no "config" key — must default to v1.
        model = ATLASModel(ATLASConfig())
        p = tmp_path / "v1_legacy.pt"
        torch.save({"model_state_dict": model.state_dict()}, p)
        atlas = ATLASInference.from_checkpoint(str(p))
        assert isinstance(atlas.model, ATLASModel)
        assert isinstance(atlas.config, ATLASConfig)

    def test_v7_round_trip(self, tmp_path) -> None:
        cfg = ATLASv7Config()
        model = ATLASModelV7(cfg)
        p = tmp_path / "v7.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, p)
        atlas = ATLASInference.from_checkpoint(str(p))
        assert isinstance(atlas.model, ATLASModelV7)
        assert isinstance(atlas.config, ATLASv7Config)
        need = atlas.config.context_len + 252
        _assert_in_bounds(_feed_and_predict(atlas, n=need + 10))

    def test_config_weights_mismatch_raises(self, tmp_path) -> None:
        model = ATLASModel(ATLASConfig())  # v1 weights
        p = tmp_path / "bad.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": ATLASv7Config()}, p)
        with pytest.raises(ValueError, match="inconsistent"):
            ATLASInference.from_checkpoint(str(p))


@pytest.mark.skipif(not os.path.exists(V7_CKPT), reason="v7 checkpoint not on disk")
class TestRealV7Checkpoint:
    def test_loads_as_v7(self) -> None:
        atlas = ATLASInference.from_checkpoint(V7_CKPT)
        assert isinstance(atlas.model, ATLASModelV7)
        assert isinstance(atlas.config, ATLASv7Config)

    def test_history_gate_then_valid_decision(self) -> None:
        atlas = ATLASInference.from_checkpoint(V7_CKPT)
        need = atlas.config.context_len + 252
        early = _feed_and_predict(atlas, n=need - 5, seed=1)
        assert early.strategy == "cash"
        assert "insufficient history" in early.reason

        atlas2 = ATLASInference.from_checkpoint(V7_CKPT)
        dec = _feed_and_predict(atlas2, n=need + 20, seed=1)
        assert "insufficient history" not in dec.reason
        _assert_in_bounds(dec)
