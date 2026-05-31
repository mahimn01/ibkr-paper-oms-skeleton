# Strategy validation report — wheel_MSFT

**Sample:** 2016-03-30 to 2026-02-24  
**Observations (T):** 2490  
**Trials searched (N):** 24  

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| Lower 95% CI on annualised Sharpe | 0.3873 | > 0.3 | ✓ |
| PBO (CSCV) | 0.5000 | < 0.5 | ✗ |
| Deflated Sharpe | 4.7329 | > 0.95 | ✓ |
| Walk-fwd 12m Sharpe > 0 | 100.0% | >= 75% | ✓ |
| MinTRL (years) | 4.8403 | <= 9.88 | ✓ |
| Cost-adjusted Sharpe | 0.7050 | > 0.3 | ✓ |

**Status:** BLOCKED
