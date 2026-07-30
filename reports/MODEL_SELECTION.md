# Model selection

**Criterion:** Lowest walk-forward RPS on common match set with acceptable calibration

**Selected model:** `ensemble_weighted`
**Walk-forward RPS:** 0.1765 (95% CI 0.1747-0.1783)

**Common match set:** 19,500 matches scored by 11 models (intersection).

## Market benchmark

Market predictions not available for comparison.

## Leaderboard (RPS)

| Rank | Model | RPS | 95% CI | Accuracy (footnote) |
|------|-------|-----|--------|---------------------|
| 1 | `ensemble_weighted` | 0.1765 | [0.1747, 0.1783] | 58.9% |
| 2 | `ensemble_avg` | 0.1768 | [0.1750, 0.1786] | 58.9% |
| 3 | `bayesian` | 0.1788 | [0.1768, 0.1809] | 58.2% |
| 4 | `poisson` | 0.1805 | [0.1782, 0.1827] | 57.9% |
| 5 | `dixon_coles` | 0.1806 | [0.1783, 0.1828] | 57.8% |
| 6 | `bivariate_poisson` | 0.1815 | [0.1792, 0.1838] | 57.5% |
| 7 | `dixon_coles_v2` | 0.1836 | [0.1819, 0.1852] | 58.1% |
| 8 | `ml` | 0.1858 | [0.1836, 0.1880] | 56.8% |
| 9 | `elo` | 0.1860 | [0.1839, 0.1882] | 57.0% |
| 10 | `baseline` | 0.2258 | [0.2244, 0.2271] | 48.0% |
| 11 | `constant` | 0.2275 | [0.2262, 0.2288] | 48.0% |

## Ties (overlapping RPS CIs)

- `ensemble_weighted, ensemble_avg, bayesian, poisson, dixon_coles, bivariate_poisson, dixon_coles_v2, ml, elo`
- `baseline, constant`

## Notes

- Selection uses walk-forward scores on the **intersection** of matches scored by all models.
- Accuracy is descriptive only; RPS is the primary metric.
- Models listed under Ties share overlapping confidence intervals: treat rank gaps inside a cluster as noise.
- If nothing beats the market, live betting should be framed as price-taker on soft markets only.
