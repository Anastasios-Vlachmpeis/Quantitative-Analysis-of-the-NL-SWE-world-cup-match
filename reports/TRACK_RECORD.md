# Live track record

**Updated (UTC):** 2026-07-30T17:36:12.742828+00:00
**Ledger rows:** 2

## Summary

| Bets | Measurable CLV | Total staked | Total P&L | Bankroll |
|---:|---:|---:|---:|---:|
| 2 | 0 | $51.00 | $+36.94 | $1036.94 |

### CLV

No measurable CLV yet. Single pre-kickoff snapshots are recorded with `measurable=False` rather than CLV=0.

## Settled bets

| # | Match | Market | Selection | Stake | P&L | CLV | Measurable |
|---:|---|---|---|---:|---:|---:|:---:|
| 1 | `6fc2ebdea7fcb9d4` | 1x2 | home | $31.00 | $+22.94 | n/a | False |
| 2 | `6fc2ebdea7fcb9d4` | totals | over_2.5 | $20.00 | $+14.00 | n/a | False |

## What this ledger records

Only the **machine-generated** bet slips produced by `live.predict`, which is
what the automated strategy actually pre-registered. Discretionary bets a human
added by hand are deliberately excluded, so the track record measures the model
rather than the operator.

This is why the NL vs SWE row set differs from `reports/BETS.md` and the P&L
figure in `reports/REPORT.md` ($36.94). That writeup settles the human bet
ticket: one Netherlands win at 1.74 for $31, plus a speculative over 2.5 for $20
that the model itself had flagged as a likely losing bet. The frozen machine slip
instead held two Netherlands win bets, at Pinnacle 1.74 and William Hill 1.73,
and no totals bet at all (totals markets were disabled in config). Both numbers
are correct for what they describe; neither restates the other.

## Artifacts

- Ledger: `data/processed/live/ledger.parquet`
- Figure: `reports/figures/live_track_record.png`
- Per-match results: `data/processed/live/results.parquet`
