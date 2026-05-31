# Strategy validation report — si_high_long_short

**Observations (T):** 1807  
**Trials searched (N):** 15  

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| Lower 95% CI on annualised Sharpe | -0.3349 | > 0.3 | ✗ |
| PBO (CSCV) | 0.3000 | < 0.5 | ✓ |
| Deflated Sharpe | 0.1113 | > 0.95 | ✗ |
| Walk-fwd 12m Sharpe > 0 | 66.9% | >= 75% | ✗ |
| MinTRL (years) | 26.2602 | <= 7.17 | ✗ |
| Cost-adjusted Sharpe | -0.2547 | > 0.3 | ✗ |

**Status:** BLOCKED
