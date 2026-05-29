"""Score a trained ATLAS policy through the hardened validation gate.

METHODOLOGY (read before trusting the verdict):
- ATLAS training reserves NO held-out test set: the env samples random 500-day
  windows across each symbol's full history, so the policy has effectively seen
  every period. There is therefore no clean out-of-sample slice available.
- This script is consequently an IN-SAMPLE NEGATIVE SCREEN, not a certification.
  The asymmetry is what makes it useful: an in-sample BLOCK is definitive (it
  can't even clear the bar on its own training distribution), while an in-sample
  pass only means "promising — needs OOS confirmation via a proper time split".
- PBO/CSCV is intentionally OMITTED: it measures overfitting of a parameter
  SELECTION across N strategy variants; a single trained RL policy has no
  in-sample config grid, so PBO does not apply. The card will therefore read
  INCOMPLETE by design (missing PBO + cost-adjusted gates). Certifying ATLAS to
  deployment standard requires retraining with a held-out time-based test set.
- DSR n_trials reflects the development search (v2..v7, many runs) so the
  significance gate is deflated for that multiple testing.

Usage:
    python scripts/validate_atlas.py \
        --checkpoint checkpoints/atlas_v7_ibkr_curriculum/atlas_v7_curriculum_final.pt \
        --episodes 50
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.inference import ATLASInference
from trading_algo.quant_core.models.atlas.train_ppo import (
    OptionsEnvironment,
    load_training_data_v2,
)
from trading_algo.quant_core.validation.report_card import build_report_card

FEATURE_DIR = "data/atlas_features_v3"  # what v7_ibkr_curriculum trained on
DEFAULT_CKPT = "checkpoints/atlas_v7_ibkr_curriculum/atlas_v7_curriculum_final.pt"


def rollout(env: OptionsEnvironment, model, n_episodes: int, device: str = "cpu"):
    """Run the greedy policy for n_episodes; return pooled daily returns + per-episode totals."""
    pooled: list[np.ndarray] = []
    ep_totals: list[float] = []
    for _ in range(n_episodes):
        obs = env.reset()
        eq0 = env._compute_equity(float(env._closes[env._t]), float(env._ivs[env._t]))
        equity = [eq0]
        while True:
            with torch.no_grad():
                action_mean, _ = model.forward_with_value(
                    obs["features"].to(device), obs["timestamps"].to(device),
                    obs["dow"].to(device), obs["month"].to(device),
                    obs["is_opex"].to(device), obs["is_qtr"].to(device),
                    obs["pre_mu"].to(device), obs["pre_sigma"].to(device),
                    obs["rtg"].to(device),
                )
            action = action_mean.squeeze(0).cpu().numpy()
            obs, _reward, done = env.step(action)
            price = float(env._closes[env._t])
            iv = float(env._ivs[env._t])
            equity.append(env._compute_equity(price, iv))
            if done:
                break
        eq = np.asarray(equity, dtype=np.float64)
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
        pooled.append(rets[np.isfinite(rets)])
        ep_totals.append((eq[-1] - eq[0]) / max(eq[0], 1e-8))
    return np.concatenate(pooled), np.asarray(ep_totals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--n-trials", type=int, default=64, help="dev-search size for DSR deflation")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    atlas = ATLASInference.from_checkpoint(args.checkpoint)
    model = atlas.model
    print(f"loaded {type(model).__name__} ({model.count_parameters():,} params) from {args.checkpoint}")

    all_data = load_training_data_v2(FEATURE_DIR, min_len=120 + 200)
    if not all_data:
        raise FileNotFoundError(f"no features under {FEATURE_DIR}")
    print(f"loaded {len(all_data)} symbols from {FEATURE_DIR}")

    # Use a v1 ATLASConfig purely for env mechanics (context_len/n_features/risk
    # params are identical across v1/v7); the v7 MODEL drives the actions.
    env = OptionsEnvironment(all_data, ATLASConfig(), regime_filter="all", reward_shaping="none")

    print(f"rolling {args.episodes} greedy episodes ...")
    rets, ep_totals = rollout(env, model, args.episodes)
    print(
        f"pooled daily returns: {rets.size}  |  "
        f"episode total return: mean {ep_totals.mean()*100:+.2f}%  "
        f"median {np.median(ep_totals)*100:+.2f}%  win-rate {float((ep_totals>0).mean())*100:.0f}%"
    )

    rc = build_report_card(
        strategy_name="atlas_v7",
        returns=rets,
        n_trials=args.n_trials,           # DSR deflation for the v2..v7 dev search
        trial_grid=None,                  # PBO N/A for a single policy (see header)
        cost_adjusted_returns=None,       # env returns already net of modelled costs
        periods_per_year=252,
        extra_warnings=[
            "IN-SAMPLE negative screen: ATLAS has no held-out test set (env samples "
            "windows across full history). Not a certification.",
            "PBO omitted: not applicable to a single RL policy (no config grid). "
            "Card is INCOMPLETE by design; deployment certification needs a held-out "
            "time-split test set.",
        ],
    )

    out = Path("validation_reports"); out.mkdir(exist_ok=True)
    (out / "atlas_v7.md").write_text(rc.render())
    print("\n" + rc.render())
    print(f"\nwrote {out/'atlas_v7.md'}")
    print(f"STATUS: {rc.status}  (in-sample screen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
