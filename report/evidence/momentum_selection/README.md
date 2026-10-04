# Momentum selection evidence

We shortlisted the five momentum configurations with the highest validation
returns, evaluated them over February 2024-January 2026, and selected **#1292**,
which achieved the highest test-period return within that shortlist.

The finalists, in validation-return order, are **1436, 1712, 1724, 1292 and 1424**.
Validation covers January 2018-December 2023. Each test portfolio starts with
$1,000,000 on January 31, 2024 and produces 24 monthly returns through
January 2026.

The [complete validation grid](../../../outputs/backtest/searches/momentum/candidates.csv.gz)
contains all 5,760 validation results and complete JSON parameter packets,
identified by `Candidate_Number`.

## Files

| Path | Contents |
|---|---|
| [evaluation_summary.csv](evaluation_summary.csv) | Validation ranks and returns, plus test returns, Sharpe ratios and drawdowns for the five finalists. |
| [evaluation_2024_2026/](evaluation_2024_2026/) | One folder per finalist, containing monthly `returns.csv`, aggregate `metrics.json` and `status.json`. |

## Reading the saved fields

- `Candidate_Number` identifies the configuration; `PctReturn_Rank` is its
  net-return rank across the complete validation grid.
- `Validation_PctReturn` and `Test_Cumulative_Return_Pct` are cumulative net
  returns in percent. `Test_Sharpe` is the annualized Sharpe ratio;
  `Test_Maximum_Drawdown_Pct` is the maximum month-end drawdown in percent.
- In each `returns.csv`, `Net` includes financing and trading costs;
  `Gross_Of_Trading_Costs` removes fees and spreads but retains financing;
  `Benchmark` is the aligned S&P 500 Total Return series. Returns are decimals.
- Each `status.json` records the candidate identifiers, completion information
  and observation count. Its `sha256` entries verify the accompanying
  `returns.csv` and `metrics.json` files.
