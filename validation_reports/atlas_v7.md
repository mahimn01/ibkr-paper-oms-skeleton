# Strategy validation report — atlas_v7

**Observations (T):** 25000  
**Trials searched (N):** 64  

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| Lower 95% CI on annualised Sharpe | 0.0283 | > 0.3 | ✗ |
| Deflated Sharpe | 0.2738 | > 0.95 | ✗ |
| Walk-fwd 12m Sharpe > 0 | 52.9% | >= 75% | ✗ |
| MinTRL (years) | 61.8896 | <= 99.21 | ✓ |

**Status:** BLOCKED

**Missing required gates:** PBO (CSCV), Cost-adjusted Sharpe

**Warnings:**
- IN-SAMPLE negative screen: ATLAS has no held-out test set (env samples windows across full history). Not a certification.
- PBO omitted: not applicable to a single RL policy (no config grid). Card is INCOMPLETE by design; deployment certification needs a held-out time-split test set.
