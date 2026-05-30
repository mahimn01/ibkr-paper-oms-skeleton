# Strategy validation report — insider_form4_opportunistic

**Observations (T):** 64  
**Trials searched (N):** 16  

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| Lower 95% CI on annualised Sharpe | -0.4687 | > 0.3 | ✗ |
| PBO (CSCV) | 0.5857 | < 0.5 | ✗ |
| Deflated Sharpe | 0.1681 | > 0.95 | ✗ |
| Walk-fwd 12m Sharpe > 0 | 52.8% | >= 75% | ✗ |
| MinTRL (years) | 11.0351 | <= 5.33 | ✗ |
| Cost-adjusted Sharpe | 0.3310 | > 0.3 | ✓ |

**Status:** BLOCKED
