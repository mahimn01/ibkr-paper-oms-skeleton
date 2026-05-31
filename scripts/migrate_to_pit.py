#!/usr/bin/env python3
"""CLI: migrate the legacy JSON cache into the bitemporal PIT store.

Usage:
    python scripts/migrate_to_pit.py \
        --cache-dir /path/to/legacy_json_cache \
        --pit-root  /path/to/pit_store \
        [--reconcile]

The script is idempotent: re-running over an already-migrated tree
will skip duplicate (symbol, date) tuples (PITStore.write_bars detects
them and emits an at-most-one warning per partition).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the top-level package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_algo.data.migration import (
    import_directory,
    reconcile,
)
from trading_algo.data.pit_store import PITStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="Legacy JSON cache directory")
    parser.add_argument("--pit-root", required=True, type=Path,
                        help="Destination PIT-store root")
    parser.add_argument("--reconcile", action="store_true",
                        help="After import, sample-reconcile close prices")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if not args.cache_dir.exists():
        print(f"[error] cache-dir does not exist: {args.cache_dir}", file=sys.stderr)
        return 2

    store = PITStore(args.pit_root)
    print(f"[info] PIT store root: {store.root}")
    print(f"[info] importing from: {args.cache_dir}")
    report = import_directory(store, args.cache_dir)
    print()
    print(report.render())

    if args.reconcile:
        print()
        print("[info] reconciling sample bars vs source files ...")
        recon = reconcile(store, args.cache_dir)
        print(f"  reconciled: {recon.reconciled_pairs}")
        print(f"  mismatched: {recon.mismatched_pairs}")
        if recon.mismatched_pairs > 0:
            return 3

    return 0 if report.files_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
