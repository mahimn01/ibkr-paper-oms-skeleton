# Strategy validation report — residual_midcap_momentum_r3000

**Observations (T):** 183  
**Trials searched (N):** 7  

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| Lower 95% CI on annualised Sharpe | -0.0796 | > 0.3 | ✗ |
| PBO (CSCV) | 0.2857 | < 0.5 | ✓ |
| Deflated Sharpe | 0.3693 | > 0.95 | ✗ |
| Walk-fwd 12m Sharpe > 0 | 61.0% | >= 75% | ✗ |
| MinTRL (years) | 22.8442 | <= 15.25 | ✗ |
| Cost-adjusted Sharpe | 0.0583 | > 0.3 | ✗ |

**Status:** BLOCKED

**Warnings:**
- SURVIVORSHIP: atlas_r3000 is a fixed ~100%-surviving set; the LONG mid-cap momentum leg is the most survivorship-inflated construction possible. Treat any positive Sharpe as optimistic and UNTRUSTWORTHY until rebuilt on a delisting-inclusive PIT universe.
- COSTS: cost-adjusted gate uses 50 bps/leg, a FLOOR for $2-20M ADV names at real size; flat-bps understates true spread+impact.
