"""Point-in-time bitemporal data layer.

Replaces the legacy JSON-cache loader (`quant_core/data/ibkr_data_loader.py`)
with a survivorship-bias-free, look-ahead-safe store backed by SQLite metadata
+ partitioned parquet bars + DuckDB query layer.

See PLAN.md §2.1 for the full schema and rationale.

Public surface:
    PITStore           - main store class; insert / query bars + metadata
    UniverseResolver   - point-in-time index membership lookups
    AdjustmentEngine   - corporate-action adjustment factors at query time
"""

from trading_algo.data.pit_store import PITStore
from trading_algo.data.universe import UniverseResolver
from trading_algo.data.corporate_actions import AdjustmentEngine

__all__ = ["PITStore", "UniverseResolver", "AdjustmentEngine"]
