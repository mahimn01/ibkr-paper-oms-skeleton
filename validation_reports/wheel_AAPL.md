# Strategy validation report — wheel_AAPL

**Sample:** 2016-03-30 to 2026-02-24  
**Observations (T):** 2490  
**Trials searched (N):** 24  

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| Lower 95% CI on annualised Sharpe | -0.0523 | > 0.3 | ✗ |
| PBO (CSCV) | 0.5714 | < 0.5 | ✗ |
| Deflated Sharpe | 0.2088 | > 0.95 | ✗ |
| Walk-fwd 12m Sharpe > 0 | 95.0% | >= 75% | ✓ |
| MinTRL (years) | 12.1806 | <= 9.88 | ✗ |
| Cost-adjusted Sharpe | -0.2134 | > 0.3 | ✗ |

**Status:** BLOCKED
