# Stock-level index tracking: a point-in-time benchmark, a long-only mandate and its attribution by sector

This repository rebuilds the S&P 500 month by month from CRSP's list of members and runs a long-only index-tracking mandate on it. It measures how far the mandate's portfolio strays from the benchmark, how well each of four covariance estimators forecasts that distance, what the turnover is made of, and where the active return comes from by sector.

The results are on the [dashboard](https://www.carlohofer.com/stock-level-tracking/dashboard/) and in the [note](https://www.carlohofer.com/stock-level-tracking/note/stock-level-tracking_note_2026-11.pdf). The project page is [carlohofer.com/projects/stock-level-tracking](https://www.carlohofer.com/projects/stock-level-tracking/).

## Findings

The study has four parts, the *benchmark*, the *tracking error*, the *turnover* and the *attribution*.

- **The benchmark.** Rebuilt from CRSP's list of index members, month by month from 1926 to 2024 and weighted by market capitalisation, it tracks CRSP's own value-weighted return of the same list within 11.5 basis points a year. Its momentum and variance spreads have correlations of 0.888 and 0.954 with Kenneth R. French's large-cap spreads.
- **The tracking error.** From July 1999 to December 2024, under a mandate with bounds per stock and sector, a turnover cap of 2 per cent a month, Energy and Utilities stocks at no more than half their benchmark weight, two excluded sub-industries and a beta of one, the four covariance estimators realise 78.9 to 83.6 basis points a year of tracking error. Every estimator's forecast underestimates it: the realised active return is 48 to 85 per cent more volatile than forecast, and on each estimator the bias statistic lies above 1.25, the upper end of the band fixed in advance, by a margin that is statistically significant at the 5 per cent level.
- **The turnover.** Index changes require 2.99 per cent a year of the Ledoit-Wolf path's 26.26 per cent of one-way turnover, i.e., 11.4 per cent. Loosening the cap from 0.5 per cent a month to none raises chosen turnover from 6.3 to 59.4 per cent a year on Ledoit-Wolf and lowers the mean forecast, while the realised tracking error changes by +3.1 basis points a year, which is not statistically significant at the 5 per cent level.
- **The attribution.** Month by month, the sector weights move the active return more than the stocks held within sectors: allocation averages 16.4 to 16.7 basis points a month in absolute value on the three structured estimators, against 10.6 to 13.8 for selection. Over the 306 months no path's mean active return, and none of its three sector terms, is statistically significant at the 5 per cent level, and the six factors of Fama and French (2015) and momentum explain 2 to 13 per cent of the active return's variance.

## The note and the dashboard

The note is a PDF with its HTML source beside it in `note/`, built by `note/build_note.py`. The dashboard, `dashboard/index.html`, is built by `dashboard/build_dashboard.py` from the same committed tables, and its figures are drawn by notebook 12 into `dashboard/figures/`.

## The notebooks

| Notebook | What it does |
|---|---|
| 01 `extraction` | Checks that the CRSP and Compustat files are the extraction the study is built on, by the hash of its manifest |
| 02 `universe` | Builds the universe month by month and counts it against CRSP's own count (G2, E2) |
| 03 `returns` | Builds each member's monthly return with delistings applied, and its market capitalisation (G3, G4) |
| 04 `reconstruction` | Builds the benchmark and sets its return beside CRSP's own return of the same list (G1, E1) |
| 05 `sectors` | Gives each member its GICS sector in force at the start of the month, and sets the start of the evaluation window |
| 06 `answer_key` | Sorts the universe on momentum and on variance and sets the spreads beside French's (E3) |
| 07 `covariance` | Estimates the four covariance matrices each month and compares their conditioning (E4, first clause) |
| 08 `mandate` | Sets out the four constraint sets and works one month through the optimiser |
| 09 `rolling_evaluation` | Computes each path's tracking error, forecast, bias statistic and active return (E4, second clause, E5, E6) |
| 10 `turnover` | Splits each path's turnover into what index changes require and what the optimiser chooses (E7) |
| 11 `attribution` | Splits the active return by sector and by factor (E8, E9) |
| 12 `figures` | Draws the four figures from the committed tables |
| 13 `paired_tests` | Tests the differences between realised tracking errors on the paired months |

`scripts/run_evaluation.py`, run after notebook 08, runs the 24 paths month by month.

## How to run

1. Obtain the extraction from Wharton Research Data Services. Each file's query and SHA-256 hash are in `results/20260928/extraction_files.csv`. CRSP corrects its history with each release (CRSP, 2025, p. 5), so the same queries run later can return other files.
2. Set the environment variable `SLT_DATA_ROOT` to the folder that holds the extraction's `manifest.json`, outside the repository. Notebook 01 stops if the manifest's hash differs from `MANIFEST_SHA256` in `src/slt/constants.py`.
3. Run `pip install -r requirements.txt`, then the notebooks in order from 01. Each writes its aggregates to `results/20260928/`, and the notebooks that need French's files download them at run time, recording each file's hash in `provenance_french.json`. After notebook 08, run `python scripts/run_evaluation.py`, which writes each path's monthly records to `outputs/evaluation/` and resumes from the last month written if it stops.
4. Notebooks 12 and 13 read only the committed tables, so they run without the extraction. Then run `python dashboard/build_dashboard.py` and `python note/build_note.py`. The note's PDF is printed by Chromium through Playwright, which is not in `requirements.txt`, and `python note/build_note.py --no-pdf` writes the HTML alone.
5. Run `python -m pytest tests`. Beside the unit tests, one test reads every file in `results/` and fails if a file names a security, carries a series of CRSP's index file or holds more rows than an aggregate needs. The tests also check that the files copied from [Optimal versus naive diversification](https://github.com/ochofer/optimal-vs-naive-diversification), i.e., the four estimators, the optimiser, the factor attribution and the paired test, are unchanged.

The notebooks are built by `build_notebooks.py` and committed without outputs.

## What is fixed in advance

Every gate, expectation, sample boundary and mandate setting is in `src/slt/constants.py`, one per line with its reason and the date it was fixed. A *gate* tests the data and stops the study if it is not met, and an *expectation* tests a result and is reported whether it is met or not. Each was written before the code that tests it ran, and the notebooks mark the comparisons added after the results were known.

## Data

The stock data are CRSP and Compustat via Wharton Research Data Services, from an extraction of 28 September 2026. They are licensed, so this repository holds none of them: it holds code, tests, the hash of the extraction's manifest and aggregate results. Nothing that reproduces CRSP's index file is committed by month, so the benchmark's return, count and total value appear only in summaries and by year. Four tables are committed by month, from July 1999 to December 2024:

- the GICS sector weights of the benchmark, each a share of the classified capitalisation
- the active returns of the twelve paths under the mandate, gross of costs
- their forecast tracking errors
- the months in which the turnover cap was raised, with the path, the cap and the turnover needed

A sector weight is committed by month only if it sums at least three securities and no security is more than 80 per cent of it, so the Unclassified sector is never committed by month. An active return is committed by month only if at least three stocks hold an active weight above one millionth and no stock contributes more than 80 per cent of it, and a cell that fails is left empty and counted. No weight of any security is committed, and nothing of the attribution by month. French's data are downloaded at run time and are not committed either.

## Terms

- **Universe**: the stocks on CRSP's list of S&P 500 members on the last trading day of the month before.
- **Benchmark**: the universe weighted by market capitalisation at the start of the month.
- **Active return**: the portfolio's return minus the benchmark's.
- **Tracking error**: the standard deviation of the monthly active return, times the square root of 12. It is forecast before the month from the covariance estimate and realised after it.
- **Bias statistic**: the standard deviation of each month's active return divided by that month's forecast. It is 1 for an accurate forecast and above 1 when the forecast is too low.
- **Mandate**: the constraints the portfolio obeys, i.e., long-only, each stock and company within 50 basis points of its benchmark weight and each sector within 2 percentage points, a turnover cap, the tilt, two excluded sub-industries and a beta of one.
- **Tilt**: every Energy and Utilities stock held at no more than half its benchmark weight.
- **Path**: one covariance estimator under one set of constraints and one turnover cap, run month by month from the benchmark.
- **Forced turnover**: the trading that index changes require. **Chosen turnover** is the rest, and the cap limits only chosen turnover.
- **Structured estimators**: the Ledoit-Wolf, factor and principal-component estimators, as against the sample covariance.
- **Allocation, selection and interaction**: the three terms of the attribution by sector of Brinson and Fachler (1985), from holding sectors at other weights than the benchmark's, from holding other stocks within a sector, and from the two together.

## References

Brinson, G. P. and Fachler, N. (1985). Measuring non-U.S. equity portfolio performance. *Journal of Portfolio Management* 11(3), 73 to 76.

CRSP (2025). *Data Descriptions Guide for CRSPAccess (FIZ) US Stock & Index Databases*. Center for Research in Security Prices, last modified 20 November 2025.

Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics* 116(1), 1 to 22.

## Other projects

**Optimal versus naive diversification.** Under an index-tracking mandate on French's 49 industries, five covariance estimators understate their own tracking error over 1979 to 2026: the realised active return is 22 to 61 per cent more volatile than forecast. The replication of DeMiguel, Garlappi and Uppal (2009) reproduces 29 of the 30 Sharpe ratios that carry a band, 26 of 27 with the value-weighted market counted once. [Page](https://www.carlohofer.com/projects/optimal-vs-naive-diversification/), [dashboard](https://www.carlohofer.com/optimal-vs-naive-diversification/dashboard/B-P_dashboard.html), [note](https://www.carlohofer.com/optimal-vs-naive-diversification/note/B-P_note_2026-10.pdf), [code](https://github.com/ochofer/optimal-vs-naive-diversification)

**Climate risk as an equity signal.** An ownership graph linking 328 ownership entities in the US and developed Europe, 302 of them listed companies with a price series, to 5,115 asset holdings, built from Global Energy Monitor's free tracker. Of 19 apparent ownership changes drawn at random and checked by hand, 12 had no corporate event behind them. [Page](https://www.carlohofer.com/projects/climate-risk-equity-signal/), [note](https://github.com/ochofer/paper1-hazard-exposure-data/releases/download/note-v1.3/what-a-vintage-difference-measures.pdf), [code](https://github.com/ochofer/paper1-hazard-exposure-data)

**Volatility-managed portfolios: a replication, the real-time critique, and the months after the sample.** Volatility estimation and position sizing on Ken French's equity factors, replicating Moreira and Muir (2017) and the real-time critique of Cederburg, O'Doherty, Wang and Yan (2020). Scaled by the inverse of last month's realised variance, a factor's Sharpe ratio is higher than the original's by a margin that is statistically significant at the 5 per cent level for momentum alone when each of the six factors is tested on its own, and for no factor after Holm's correction for the six tests. In the 136 months after Moreira and Muir's sample no factor's difference is statistically significant, and the intervals are too wide to rule out the differences found on their sample. [Page](https://www.carlohofer.com/projects/volatility-managed-portfolios/), [dashboard](https://www.carlohofer.com/volatility-managed-portfolios/dashboard/B-Q_dashboard.html), [note](https://www.carlohofer.com/volatility-managed-portfolios/note/B-Q_note_2026-11.pdf), [code](https://github.com/ochofer/volatility-managed-portfolios)

**Rules-based portfolio.** Runs a real-money portfolio under rules published before the first order, with every later change dated in the repository: two UCITS ETFs at 70/30, a monthly cycle with pre-trade checks, look-through exposure and ex-ante risk on a public dashboard. [Page](https://www.carlohofer.com/projects/rules-based-portfolio/), [dashboard](https://www.carlohofer.com/rules-based-portfolio/), [guide](https://www.carlohofer.com/rules-based-portfolio/guide/guide.pdf), [code](https://github.com/ochofer/rules-based-portfolio)
