# Strategy validation report — gld_overnight_drift

**Sample:** 2015-12-31 to 2026-02-23  
**Observations (T):** 2550  
**Trials searched (N):** 3  

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| Lower 95% CI on annualised Sharpe | 0.1914 | > 0.3 | ✗ |
| PBO (CSCV) | 0.0000 | < 0.5 | ✓ |
| Deflated Sharpe | 0.8997 | > 0.95 | ✗ |
| Walk-fwd 12m Sharpe > 0 | 59.3% | >= 75% | ✗ |
| MinTRL (years) | 4.8679 | <= 10.12 | ✓ |
| Cost-adjusted Sharpe | 0.5436 | > 0.3 | ✓ |

**Status:** BLOCKED
