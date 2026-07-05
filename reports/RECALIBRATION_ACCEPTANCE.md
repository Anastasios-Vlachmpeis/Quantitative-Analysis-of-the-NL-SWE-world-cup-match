# Plan 14 recalibration acceptance

Walk-forward refit_every=10 (v1 RPS from existing predictions_dixon_coles at refit_every=10).

## Totals reliability (over 2.5)

Figure: [`figures/totals_reliability_recalibration.png`](figures/totals_reliability_recalibration.png)

| Model | ECE |
|---|---|
| dixon_coles | 0.1010 |
| dixon_coles_v2_raw | 0.0462 |
| dixon_coles_v2 | 0.0462 |

## 1X2 RPS (common match set)

| Model | RPS | CI |
|---|---|---|
| dixon_coles | 0.1813 | [0.1791, 0.1835] |
| dixon_coles_v2 | 0.1840 | [0.1824, 0.1856] |

## Draw rate (aggregate)

| Model | Pred draw | Actual draw | Gap (pp) |
|---|---|---|---|
| dixon_coles | 0.249 | 0.232 | +1.7 |
| dixon_coles_v2 | 0.246 | 0.232 | +1.4 |

## NL-SWE spot anchor (over 2.5)

| Model | Raw P(over) | Cal P(over) | Market | Gap raw (pp) | Gap cal (pp) |
|---|---|---|---|---|---|
| dixon_coles | 0.714 | 0.714 | 0.560 | +15.4 | +15.4 |
| dixon_coles_v2 | 0.684 | 0.684 | 0.560 | +12.4 | +12.4 |
