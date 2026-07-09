"""Regression and contract tests for IBKR account/portfolio CLI commands."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

import pytest
from ib_async import StartupFetch, StartupFetchNONE, util

from trading_algo import ibkr_tool


def _args(**overrides: Any) -> argparse.Namespace:
    defaults = {
        "account": None,
        "client_id": 177,
        "format": "json",
        "host": None,
        "market_data_type": None,
        "port": None,
        "symbol": None,
        "tag": None,
        "tags": None,
        "timeout": 5.0,
        "wait": 0.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _contract(
    symbol: str,
    *,
    con_id: int,
    sec_type: str = "STK",
    local_symbol: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        conId=con_id,
        secType=sec_type,
        symbol=symbol,
        localSymbol=local_symbol or symbol,
        currency="USD",
        exchange="SMART",
        lastTradeDateOrContractMonth="",
        right="",
        strike=0.0,
        multiplier="100" if sec_type == "OPT" else "",
    )


def _portfolio_item(
    account: str,
    symbol: str,
    *,
    con_id: int,
    market_value: float,
    realized_pnl: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        account=account,
        contract=_contract(symbol, con_id=con_id),
        position=10.0,
        marketPrice=market_value / 10.0,
        marketValue=market_value,
        averageCost=9.0,
        unrealizedPNL=market_value - 90.0,
        realizedPNL=realized_pnl,
    )


class _FakeIB:
    def __init__(
        self,
        *,
        accounts: list[str],
        account_values: list[Any] | None = None,
        portfolio_items: list[Any] | None = None,
        positions: list[Any] | None = None,
        pnl_by_account: dict[str, Any] | None = None,
        pnl_single: Any | None = None,
    ) -> None:
        self._accounts = accounts
        self._account_values = account_values or []
        self._portfolio_items = portfolio_items or []
        self._positions = positions or []
        self._pnl_by_account = pnl_by_account or {}
        self._pnl_single = pnl_single
        self.connected = True
        self.disconnect_calls = 0
        self.sleep_calls: list[float] = []
        self.cancelled_pnl: list[str] = []
        self.cancelled_single: list[tuple[str, str, int]] = []

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def managedAccounts(self) -> list[str]:
        return list(self._accounts)

    def accountSummary(self, account: str = "") -> list[Any]:
        return []

    def accountValues(self) -> list[Any]:
        return list(self._account_values)

    def portfolio(self) -> list[Any]:
        return list(self._portfolio_items)

    def positions(self) -> list[Any]:
        return list(self._positions)

    def reqPnL(self, account: str, model_code: str) -> Any:
        assert model_code == ""
        return self._pnl_by_account[account]

    def cancelPnL(self, account: str, model_code: str) -> None:
        assert model_code == ""
        self.cancelled_pnl.append(account)

    def reqPnLSingle(self, account: str, model_code: str, con_id: int) -> Any:
        assert model_code == ""
        assert self._pnl_single is not None
        return self._pnl_single

    def cancelPnLSingle(self, account: str, model_code: str, con_id: int) -> None:
        self.cancelled_single.append((account, model_code, con_id))

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)


class _ConnectFactory:
    def __init__(self, sessions: list[_FakeIB]) -> None:
        self.sessions = sessions
        self.calls: list[dict[str, Any]] = []

    def __call__(self, _args: argparse.Namespace, **kwargs: Any) -> _FakeIB:
        self.calls.append(kwargs)
        if not self.sessions:
            raise AssertionError("unexpected extra IB connection")
        return self.sessions.pop(0)


def test_connect_forwards_readonly_account_and_sync_controls(monkeypatch) -> None:
    class FakeConnector:
        def __init__(self) -> None:
            self.connect_call: tuple[tuple[Any, ...], dict[str, Any]] | None = None

        def connect(self, *args: Any, **kwargs: Any) -> None:
            self.connect_call = (args, kwargs)

        def reqMarketDataType(self, _kind: int) -> None:
            raise AssertionError("market data type should not be requested")

    connector = FakeConnector()
    monkeypatch.setattr(ibkr_tool, "IB", lambda: connector)

    result = ibkr_tool._connect(
        _args(),
        readonly=True,
        account="A1",
        raise_sync_errors=True,
        fetch_fields=StartupFetch.ACCOUNT_UPDATES,
    )

    assert result is connector
    assert connector.connect_call == (
        (ibkr_tool.DEFAULT_HOST, ibkr_tool.DEFAULT_PORT),
        {
            "clientId": 177,
            "timeout": 5.0,
            "readonly": True,
            "account": "A1",
            "raiseSyncErrors": True,
            "fetchFields": StartupFetch.ACCOUNT_UPDATES,
        },
    )


def test_portfolio_multi_account_snapshot_is_complete_sorted_and_readonly(
    monkeypatch, capsys
) -> None:
    discovery = _FakeIB(accounts=["B2", "A1"])
    account_a = _FakeIB(
        accounts=["A1", "B2"],
        portfolio_items=[
            _portfolio_item("B2", "IGNORE", con_id=99, market_value=1.0),
            _portfolio_item(
                "A1",
                "ZZZ",
                con_id=2,
                market_value=120.0,
                realized_pnl=util.UNSET_DOUBLE,
            ),
            _portfolio_item("A1", "AAA", con_id=1, market_value=100.0),
        ],
    )
    account_b = _FakeIB(
        accounts=["A1", "B2"],
        portfolio_items=[_portfolio_item("B2", "BBB", con_id=3, market_value=80.0)],
    )
    factory = _ConnectFactory([discovery, account_a, account_b])
    monkeypatch.setattr(ibkr_tool, "_connect", factory)

    assert ibkr_tool.cmd_portfolio(_args()) == 0

    rows = json.loads(capsys.readouterr().out)
    assert [(row["account"], row["symbol"]) for row in rows] == [
        ("A1", "AAA"),
        ("A1", "ZZZ"),
        ("B2", "BBB"),
    ]
    assert rows[1]["realizedPNL"] is None
    assert [call.get("account", "") for call in factory.calls] == ["", "A1", "B2"]
    assert factory.calls[0] == {
        "readonly": True,
        "account": "",
        "raise_sync_errors": False,
        "fetch_fields": StartupFetchNONE,
    }
    for call in factory.calls[1:]:
        assert call["readonly"] is True
        assert call["raise_sync_errors"] is True
        assert call["fetch_fields"] == StartupFetch.ACCOUNT_UPDATES
    assert all(
        session.disconnect_calls == 1 for session in (discovery, account_a, account_b)
    )


def test_values_uses_account_scoped_startup_snapshot_and_casefold_filter(
    monkeypatch, capsys
) -> None:
    discovery = _FakeIB(accounts=["A1"])
    snapshot = _FakeIB(
        accounts=["A1"],
        account_values=[
            SimpleNamespace(
                account="A1",
                tag="NetLiquidation",
                value="100",
                currency="USD",
                modelCode="",
            ),
            SimpleNamespace(
                account="A1",
                tag="BuyingPower",
                value="50",
                currency="USD",
                modelCode="",
            ),
        ],
    )
    factory = _ConnectFactory([discovery, snapshot])
    monkeypatch.setattr(ibkr_tool, "_connect", factory)

    assert ibkr_tool.cmd_values(_args(account="A1", tag="liQUID")) == 0

    assert json.loads(capsys.readouterr().out) == [
        {
            "account": "A1",
            "tag": "NetLiquidation",
            "value": "100",
            "currency": "USD",
            "modelCode": "",
        }
    ]
    assert factory.calls[1]["account"] == "A1"
    assert snapshot.disconnect_calls == 1


def test_positions_reports_cost_basis_not_false_market_value(
    monkeypatch, capsys
) -> None:
    fake = _FakeIB(
        accounts=["A1"],
        positions=[
            SimpleNamespace(
                account="A1",
                contract=_contract("XYZ", con_id=1),
                position=-4.0,
                avgCost=12.5,
            )
        ],
    )
    monkeypatch.setattr(ibkr_tool, "_connect", lambda *_a, **_k: fake)

    assert ibkr_tool.cmd_positions(_args(symbol="xyz")) == 0

    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["costBasis"] == -50.0
    assert "marketValue" not in rows[0]
    assert fake.disconnect_calls == 1


def test_unknown_account_fails_validation_and_disconnects(monkeypatch) -> None:
    fake = _FakeIB(accounts=["A1"])
    monkeypatch.setattr(ibkr_tool, "_connect", lambda *_a, **_k: fake)

    with pytest.raises(ValueError, match="Unknown IBKR account"):
        ibkr_tool.cmd_positions(_args(account="MISSING"))

    assert fake.disconnect_calls == 1


def test_pnl_subscribes_concurrently_normalizes_unset_and_always_cancels(
    monkeypatch, capsys
) -> None:
    fake = _FakeIB(
        accounts=["B2", "A1"],
        pnl_by_account={
            "A1": SimpleNamespace(
                dailyPnL=1.0, unrealizedPnL=2.0, realizedPnL=util.UNSET_DOUBLE
            ),
            "B2": SimpleNamespace(dailyPnL=3.0, unrealizedPnL=4.0, realizedPnL=5.0),
        },
    )
    monkeypatch.setattr(ibkr_tool, "_connect", lambda *_a, **_k: fake)

    assert ibkr_tool.cmd_pnl(_args(wait=0.25)) == 0

    rows = json.loads(capsys.readouterr().out)
    assert [row["account"] for row in rows] == ["A1", "B2"]
    assert rows[0]["realizedPnL"] is None
    assert fake.sleep_calls == []  # already populated; no fixed-delay penalty
    assert fake.cancelled_pnl == ["A1", "B2"]
    assert fake.disconnect_calls == 1


def test_pnl_cancels_and_disconnects_when_output_fails(monkeypatch) -> None:
    fake = _FakeIB(
        accounts=["A1"],
        pnl_by_account={
            "A1": SimpleNamespace(dailyPnL=1.0, unrealizedPnL=2.0, realizedPnL=3.0)
        },
    )
    monkeypatch.setattr(ibkr_tool, "_connect", lambda *_a, **_k: fake)
    monkeypatch.setattr(
        ibkr_tool,
        "_emit",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("stdout failed")),
    )

    with pytest.raises(OSError, match="stdout failed"):
        ibkr_tool.cmd_pnl(_args())

    assert fake.cancelled_pnl == ["A1"]
    assert fake.disconnect_calls == 1


def test_pnl_timeout_never_emits_constructor_defaults(monkeypatch) -> None:
    fake = _FakeIB(
        accounts=["A1"],
        pnl_by_account={
            "A1": SimpleNamespace(
                dailyPnL=float("nan"),
                unrealizedPnL=float("nan"),
                realizedPnL=float("nan"),
            )
        },
    )
    monkeypatch.setattr(ibkr_tool, "_connect", lambda *_a, **_k: fake)

    with pytest.raises(TimeoutError, match="initial account PnL update"):
        ibkr_tool.cmd_pnl(_args(wait=0.0))

    assert fake.cancelled_pnl == ["A1"]
    assert fake.disconnect_calls == 1


def test_pnl_single_normalizes_unset_and_cleans_up(monkeypatch, capsys) -> None:
    fake = _FakeIB(
        accounts=["A1"],
        pnl_single=SimpleNamespace(
            position=2.0,
            dailyPnL=1.0,
            unrealizedPnL=2.0,
            realizedPnL=util.UNSET_DOUBLE,
            value=util.UNSET_DOUBLE,
        ),
    )
    monkeypatch.setattr(ibkr_tool, "_connect", lambda *_a, **_k: fake)

    assert ibkr_tool.cmd_pnl_single(_args(account="A1", con_id=42)) == 0

    row = json.loads(capsys.readouterr().out)
    assert row["realizedPnL"] is None
    assert row["value"] is None
    assert fake.cancelled_single == [("A1", "", 42)]
    assert fake.disconnect_calls == 1


def test_pnl_single_timeout_does_not_report_default_flat_position(monkeypatch) -> None:
    fake = _FakeIB(
        accounts=["A1"],
        pnl_single=SimpleNamespace(
            position=0,
            dailyPnL=float("nan"),
            unrealizedPnL=float("nan"),
            realizedPnL=float("nan"),
            value=float("nan"),
        ),
    )
    monkeypatch.setattr(ibkr_tool, "_connect", lambda *_a, **_k: fake)

    with pytest.raises(TimeoutError, match="initial position PnL update"):
        ibkr_tool.cmd_pnl_single(_args(account="A1", con_id=42, wait=0.0))

    assert fake.cancelled_single == [("A1", "", 42)]
    assert fake.disconnect_calls == 1


def test_portfolio_main_end_to_end_writes_success_audit(
    monkeypatch, tmp_path, capsys
) -> None:
    discovery = _FakeIB(accounts=["A1"])
    snapshot = _FakeIB(
        accounts=["A1"],
        portfolio_items=[_portfolio_item("A1", "ABC", con_id=7, market_value=123.0)],
    )
    factory = _ConnectFactory([discovery, snapshot])
    monkeypatch.setattr(ibkr_tool, "_connect", factory)
    monkeypatch.setenv("TRADING_AUDIT_DIR", str(tmp_path / "audit"))

    rc = ibkr_tool.main(
        [
            "portfolio",
            "--account",
            "A1",
            "--client-id",
            "991",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)[0]["symbol"] == "ABC"
    audit_files = list((tmp_path / "audit").glob("*.jsonl"))
    assert len(audit_files) == 1
    audit_entry = json.loads(audit_files[0].read_text().strip())
    assert audit_entry["cmd"] == "portfolio"
    assert audit_entry["exit_code"] == 0
    assert audit_entry["args"]["client_id"] == 991
