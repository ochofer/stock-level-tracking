# Stock-level index tracking: a point-in-time benchmark, a long-only mandate and its attribution by sector

The study rebuilds the S&P 500 month by month from the list of its members kept by the Center for Research in Security Prices (CRSP), and runs a long-only index-tracking mandate on it. It measures how far the mandate's portfolio strays from the benchmark, how well each of four covariance estimators forecasts that distance, what the turnover is made of, and where the active return comes from by sector.

The results are on the [dashboard](https://www.carlohofer.com/stock-level-tracking/dashboard/) and in the [note](https://www.carlohofer.com/stock-level-tracking/note/stock-level-tracking_note_2026-11.pdf). The project page is [carlohofer.com/projects/stock-level-tracking](https://www.carlohofer.com/projects/stock-level-tracking/).

## Terms

- **Universe**: the stocks on CRSP's list of S&P 500 members on the last trading day of the month before.
- **Benchmark**: the universe weighted by market capitalisation at the start of the month.
- **Value-weighted**: weighted by market capitalisation, as the benchmark is.
- **GICS**: the Global Industry Classification Standard, which assigns each company a sector and, within it, an industry group, an industry and a sub-industry.
- **Active return**: the portfolio's return minus the benchmark's.
- **Tracking error**: the standard deviation of the monthly active return, times the square root of 12. It is forecast before the month from the covariance estimate and realised after it.
- **Covariance estimators**: the four ways the study estimates the stocks' covariance matrix from three years of daily returns. The sample estimator is the sample covariance itself, the Ledoit-Wolf estimator shrinks it towards a scaled identity matrix (Ledoit and Wolf, 2004), the factor estimator is a factor model on Kenneth R. French's five factors and momentum, and the principal-component estimator is a model with the benchmark as its first factor and four further components.
- **Structured estimators**: the Ledoit-Wolf, factor and principal-component estimators, as against the sample estimator.
- **Bias statistic**: the standard deviation of each month's active return divided by that month's forecast. It is 1 for an accurate forecast and above 1 when the forecast is too low.
- **Band**: the range of the bias statistic within which a forecast counts as accurate, 0.8 to 1.25.
- **Mandate**: the constraints the portfolio obeys, i.e., long-only, each stock and company outside the tilted sectors and the excluded sub-industries within 50 basis points of its benchmark weight and each untilted sector within 2 percentage points, a turnover cap, the tilt, two excluded sub-industries and a beta of one.
- **Tilt**: every Energy and Utilities stock held at no more than half its benchmark weight.
- **Optimised portfolio**: the portfolio the optimiser builds month by month from the benchmark with one covariance estimator, under one constraint set and one turnover cap. Each is named by its estimator, as in "the optimised portfolio built with Ledoit-Wolf", and a result given for an estimator is that of the optimised portfolio built with it.
- **Constraint set**: one of the four cumulative sets of the mandate's constraints, from long-only alone to the full mandate.
- **Evaluation window**: the 306 months from July 1999 to December 2024 over which the optimised portfolios run.
- **Forced turnover**: the trading that index changes require. **Chosen turnover** is the rest, and the turnover cap limits only chosen turnover.
- **Circular block bootstrap**: a bootstrap that builds each new sample from blocks of 12 consecutive months, so that months close in time stay together, a block that runs past the last month continuing from the first.
- **Studentised interval**: a bootstrap interval in which each resample's distance from the estimate is divided by that resample's own standard error.
- **Allocation, selection and interaction**: the three terms of the attribution by sector of Brinson and Fachler (1985), from holding sectors at other weights than the benchmark's, from holding other stocks within a sector, and from the two together.

## Findings

The study has four parts, the *benchmark*, the *tracking error*, the *turnover* and the *attribution*.

- **The benchmark.** Rebuilt from CRSP's list of index members, month by month from 1926 to 2024 and weighted by market capitalisation, it tracks CRSP's own value-weighted return of the same list within 11.5 basis points a year. Its spreads between the top and bottom quintiles on momentum and on variance have correlations of 0.888 from 1927 and 0.954 from April 1990 with the same spreads within Kenneth R. French's largest size quintile.
- **The tracking error.** From July 1999 to December 2024, under a mandate with bounds per stock and sector, a turnover cap of 2 per cent a month, Energy and Utilities stocks at no more than half their benchmark weight, two excluded sub-industries and a beta of one, the optimised portfolios built with the four covariance estimators realise 78.9 to 83.6 basis points a year of tracking error. Every estimator's forecast underestimates it: the realised active return is 48 to 85 per cent more volatile than forecast. On each estimator the bias statistic lies above 1.25, the upper end of the band, by a margin that is statistically significant at the 5 per cent level on a circular block bootstrap of the months. For Ledoit-Wolf the margin is narrow: on a studentised interval at blocks of 24 months its lower end is 1.22, so there its excess over 1.25 is not statistically significant, and the data do not rule out a forecast inside the band.
- **The turnover.** Index changes require 2.99 per cent a year of the 26.26 per cent of one-way turnover of the optimised portfolio built with Ledoit-Wolf, i.e., 11.4 per cent. Loosening the turnover cap from 0.5 per cent a month to none raises chosen turnover from 6.3 to 59.4 per cent a year on Ledoit-Wolf and lowers the mean forecast, while the realised tracking error changes by +3.1 basis points a year, which is not statistically significant at the 5 per cent level, and the comparison could detect a change of 9.1 basis points a year four times in five.
- **The attribution.** In mean absolute monthly terms, the sector weights move the active return more than the stocks held within sectors: allocation averages 16.4 to 16.7 basis points a month in absolute value on the three structured estimators, against 10.6 to 13.8 for selection. Over the 306 months no optimised portfolio's mean active return, and none of its three sector terms, is statistically significant at the 5 per cent level: a mean active return of about 35 basis points a year in either direction would be needed. The five factors of Fama and French (2015) and momentum explain 2 to 13 per cent of the active return's variance.

## Gates and expectations

Every gate, expectation, sample boundary and mandate setting is in `src/slt/constants.py`, one per line with its reason and its date. A *gate* tests the data and stops the study if it is not met, and an *expectation* tests a result and is reported whether it is met or not. The study has four gates, G1 to G4, and nine expectations, E1 to E9. The 25 thresholds of the gates and expectations were written before the code that tests them ran.

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
| 09 `rolling_evaluation` | Computes each optimised portfolio's tracking error, forecast, bias statistic and active return (E4, second clause, E5, E6) |
| 10 `turnover` | Splits each optimised portfolio's turnover into what index changes require and what the optimiser chooses (E7) |
| 11 `attribution` | Splits the active return by sector and by factor (E8, E9) |
| 12 `figures` | Draws the four figures from the committed tables |
| 13 `paired_tests` | Tests the differences between realised tracking errors on the paired months |

`scripts/run_evaluation.py`, run after notebook 08, runs the 24 optimised portfolios month by month.

## How to run

1. Obtain the extraction from Wharton Research Data Services. Each file's query and SHA-256 hash are in `results/20260928/extraction_files.csv`. CRSP revises its history with each release (CRSP, 2025, p. 5), so the same queries run later can return other files.
2. Set the environment variable `SLT_DATA_ROOT` to the folder that holds the extraction's `manifest.json`, outside the repository. Notebook 01 stops if the manifest's hash differs from `MANIFEST_SHA256` in `src/slt/constants.py`.
3. Run `pip install -r requirements.txt`, then the notebooks in order from 01. Each writes its aggregates to `results/20260928/`, and the notebooks that need French's files download them at run time, recording each file's hash in `provenance_french.json`. After notebook 08, run `python scripts/run_evaluation.py`, which writes each optimised portfolio's monthly records to `outputs/evaluation/` and resumes from the last month written if it stops.
4. Notebooks 12 and 13 read only the committed tables, so they run without the extraction. Then run `python dashboard/build_dashboard.py` and `python note/build_note.py`. The note's PDF is printed by Chromium through Playwright, which is not in `requirements.txt`, and `python note/build_note.py --no-pdf` writes the HTML alone.
5. Run `python -m pytest tests`. Beside the unit tests, one test reads every file in `results/` and fails if a file names a security, carries a series of CRSP's index file or holds more rows than an aggregate needs. The tests also check that the files copied from [Optimal versus naive diversification](https://github.com/ochofer/optimal-vs-naive-diversification), i.e., the four estimators, the optimiser, the factor attribution and the paired test, are unchanged.

The notebooks are built by `build_notebooks.py` and committed without outputs.

## Data

The stock data are CRSP and Compustat via Wharton Research Data Services, from an extraction of 28 September 2026 and, for the GICS history that gives the sectors, of 10 November 2026. They are licensed, so this repository holds none of them: it holds code, tests, the hash of the extraction's manifest and aggregate results. Nothing that reproduces CRSP's index file is committed by month, so the benchmark's return, count and total value appear only in summaries and by year. Five tables are committed by month, from July 1999 to December 2024:

- the GICS sector weights of the benchmark, each a share of the classified capitalisation
- the active returns of the twelve optimised portfolios under the mandate, gross of costs
- their forecast tracking errors
- the allocation and the selection of the four optimised portfolios at the 2 per cent cap, each summed over the twelve sectors
- the months in which the turnover cap was raised, with the optimised portfolio, the cap and the turnover needed

A sector weight is committed by month only if it sums at least three securities and no security is more than 80 per cent of it, so the Unclassified sector, i.e., the members with no GICS sector, is never committed by month. An active return is committed by month only if at least three stocks hold an active weight above one millionth and no stock contributes more than 80 per cent of it, and a cell that fails is left empty and counted. The forecasts, the allocation and the selection are left empty where the active return is. No weight of any security is committed, and no attribution term of a single sector by month. French's data are downloaded at run time and are not committed either.

## References

Brinson, G. P. and Fachler, N. (1985). Measuring non-U.S. equity portfolio performance. *Journal of Portfolio Management* 11(3), 73 to 76.

CRSP (2025). *Data Descriptions Guide for CRSPAccess (FIZ) US Stock & Index Databases*. Center for Research in Security Prices, last modified 20 November 2025.

Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics* 116(1), 1 to 22.

Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2), 365 to 411.

## Other projects

**Optimal versus naive diversification: a replication, and the same question under an index mandate.** Replicates DeMiguel, Garlappi and Uppal (2009): 29 of the 30 Sharpe ratios tested are within 0.03 of the published values, and all 26 turnover figures of the portfolios that trade are within 25 per cent. It then asks the paper's question under an index mandate on Ken French's 49 US industries. Applying the mandate's tilt without an optimiser realises 105.6 basis points a year of tracking error, 14.7 to 25.9 more than the optimised portfolios with the same tilt, each difference statistically significant at the 5 per cent level after Holm's correction. [Page](https://www.carlohofer.com/projects/optimal-vs-naive-diversification/), [dashboard](https://www.carlohofer.com/optimal-vs-naive-diversification/dashboard/B-P_dashboard.html), [note](https://www.carlohofer.com/optimal-vs-naive-diversification/note/B-P_note_2026-11.pdf), [code](https://github.com/ochofer/optimal-vs-naive-diversification)

**Which risk model for an index mandate? Covariance estimators and tracking error on 49 US industries.** Runs a long-only index-tracking mandate on Ken French's 49 US industries from 1979 to 2026 with six covariance estimators. The factor and principal-component models realise 7 to 9 basis points a year more tracking error than the sample covariance's 84.7, each difference statistically significant at the 5 per cent level after Holm's correction. On minimum-variance portfolios, by which Dom, Howard, Jansen and Lohre (2025) judge estimators, the same two models realise 0.95 to 1.58 percentage points a year more volatility when weights may be negative, and within 0.06 points of the sample covariance when they may not. [Page](https://www.carlohofer.com/projects/risk-models-for-index-tracking/), [dashboard](https://www.carlohofer.com/risk-models-for-index-tracking/dashboard/), [note](https://www.carlohofer.com/risk-models-for-index-tracking/note/risk-models-for-index-tracking_note_2026-11.pdf), [code](https://github.com/ochofer/risk-models-for-index-tracking)

**Climate risk as an equity signal.** An ownership graph linking 328 ownership entities in the US and developed Europe, 302 of them listed companies with a price series, to 5,115 asset holdings, built from Global Energy Monitor's free tracker. Of 19 apparent ownership changes drawn at random and checked by hand, 12 had no corporate event behind them. [Page](https://www.carlohofer.com/projects/climate-risk-equity-signal/), [note](https://github.com/ochofer/paper1-hazard-exposure-data/releases/latest/download/what-a-vintage-difference-measures.pdf), [code](https://github.com/ochofer/paper1-hazard-exposure-data)

**Volatility-managed portfolios: a replication, the real-time critique, and the months after the sample.** Volatility estimation and position sizing on Ken French's six equity factors, replicating Moreira and Muir (2017) and the real-time critique of Cederburg, O'Doherty, Wang and Yan (2020). A volatility-managed factor is the factor scaled each month by the inverse of last month's realised variance. Its Sharpe ratio is higher than the original factor's by a margin that is statistically significant at the 5 per cent level for momentum alone when each of the six factors is tested on its own, and for no factor after Holm's correction for the six tests. In the 136 months after Moreira and Muir's sample no factor's difference is statistically significant, and for the six volatility-managed factors the confidence intervals are too wide to rule out the differences found on their sample. [Page](https://www.carlohofer.com/projects/volatility-managed-portfolios/), [dashboard](https://www.carlohofer.com/volatility-managed-portfolios/dashboard/B-Q_dashboard.html), [note](https://www.carlohofer.com/volatility-managed-portfolios/note/B-Q_note_2026-11.pdf), [code](https://github.com/ochofer/volatility-managed-portfolios)

**Return prediction: machine learning on the S&P 500's members, corrected for overfitting.** Forecasts of each S&P 500 member's return over the next month from twelve characteristics built from CRSP and Compustat, by six models from least squares to a neural network, out of sample from 2006 to 2024. The study follows the method of Gu, Kelly and Xiu (2020), reduced to the index's members, and corrects the highest Sharpe ratio among the portfolios built from these forecasts for overfitting, that is, for having tried many versions and kept the one with the highest. In build: its findings are stated when the study is complete. [Page](https://www.carlohofer.com/projects/ml-return-prediction-backtest-overfitting/), [code](https://github.com/ochofer/ml-return-prediction-backtest-overfitting)

**Rules-based portfolio.** Runs a real-money portfolio under rules published before the first order, with every later change dated in the repository: two UCITS ETFs at 70/30, a monthly cycle with each rule checked before the orders, and a public dashboard with the look-through of both ETFs and the estimated risk. [Page](https://www.carlohofer.com/projects/rules-based-portfolio/), [dashboard](https://www.carlohofer.com/rules-based-portfolio/), [guide](https://www.carlohofer.com/rules-based-portfolio/guide/guide.pdf), [code](https://github.com/ochofer/rules-based-portfolio)
