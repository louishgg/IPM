# Competition rules and behavioral questionnaire

Investments & Portfolio Management - FINA0053-1

This file retains the competition rules and questionnaire used in the report.
The rules are transcribed from the course competition materials, pages 1-2;
the questionnaire and calibration formulas are from pages 7-8.

## Competition rules

- Competition period: February 13, 2026 through May 6, 2026.
- Initial cash balance: $1,000,000.
- Benchmark: S&P 500 Total Return.
- Allowed securities: S&P 500 stocks only.
- Portfolio size: at least 20 distinct tickers, including at least 10 long
  and 10 short positions at all times.
- Short selling and margin are allowed; maximum gross exposure is 200%.
- Cash earns 2% annually, which is also the risk-free rate for risk-adjusted
  performance measures.
- Loans cost 8% annually.
- Execution uses prevailing bid and ask prices, with a fixed $2 fee per trade.
  The bid-ask spread is an implicit transaction cost embedded in execution.

## Behavioral questionnaire

Agreement scale: **1 = Strongly disagree; 7 = Strongly agree**.
The original first-person wording of the questions is preserved below.

1. I feel comfortable seeing my portfolio fluctuate by more than 10% in a single month.
2. I prefer a guaranteed small gain to a risky investment with a higher expected return.
3. When markets fall sharply, I tend to sell part of my holdings to reduce losses.
4. I am confident that I can identify undervalued stocks better than the average investor.
5. Short-term volatility does not affect my long-term investment decisions.
6. I consider risk as an opportunity rather than a threat.
7. I would increase my equity exposure after a market downturn if fundamentals remain unchanged.
8. I worry frequently about market news and check prices several times per day.
9. I would rather miss a gain than suffer a loss.
10. I enjoy making my own investment decisions without following popular opinion.

## Calibration formulas

Reverse-score items 2, 3, 8 and 9 as $Q_i^*=8-Q_i$; keep the original scores
for all other items. Then compute

$$
\mathrm{BRTI}=\frac{1}{10}\sum_{i=1}^{10}Q_i^*,\qquad
q=\frac{\mathrm{BRTI}-1}{6},\qquad
\lambda=12.5\exp[-0.35(\mathrm{BRTI}-1)].
$$

The mean-variance utility function is

$$
U=\mathbb{E}[R_p]-\frac{\lambda}{2}\sigma_p^2.
$$

The report and saved frontier use these formulas. The responses and their
detailed scoring appear in the report's Objective and risk preference section.
