# Stock-Level Tracking: index tracking and attribution at the stock level

This study is in build and at milestone 1 of 4, the data layer.

I build a point-in-time S&P 500 universe from CRSP and Compustat and check it in three ways: against CRSP's own return, count and total value for the same universe, and against Ken French's portfolios formed on size and momentum and on size and variance. On that universe I then run a long-only mandate that holds close to the benchmark, under four covariance estimators from 1993 to 2024, split its turnover into the part that changes in index membership force and the part the optimiser chooses, and attribute its active return by sector with the method of Brinson and Fachler (1985).

## What is pre-committed

Every gate, expectation, sample boundary and mandate level is in `src/slt/constants.py`, one per line with its reason and the date it was fixed, before any code that judges it has run. A gate that is not met stops the study, and the data layer is then published alone with the reason. An expectation that is not met is reported as a finding, in the same words as one that is met.

## Data

The stock data are CRSP and Compustat, via Wharton Research Data Services, accessed through Queen Mary University of London, from an extraction of 28 September 2026. They are licensed, and this repository holds none of them: it holds code, tests, the SHA-256 of the extraction's manifest and aggregate results. Ken French's data library is downloaded at run time and is not committed either.

## Terms

- **universe**: the point-in-time S&P 500, that is, the stocks in CRSP's list of index members on the last trading day of the month before.
- **benchmark**: the cap-weighted universe, in which each stock's weight is its market capitalisation at the start of the month divided by the universe's total.
- **data layer**: the universe with its returns, delisting returns, market capitalisations and sectors, together with the checks run on it before the study uses it.
- **gate**: a test of the data layer fixed in advance. A gate that is not met stops the study.
- **expectation**: a test of a result fixed in advance. An expectation that is not met is reported as a finding.
- **active return**: the portfolio's return minus the benchmark's in the same month.
- **tracking error**: the standard deviation of the monthly active return, times the square root of 12.
- **mandate**: the full set of constraints the portfolio obeys. It is long-only, holds each stock within 50 basis points of its benchmark weight and each sector within 2 percentage points, caps chosen turnover, applies the tilt, excludes two sub-industries and keeps a beta of one to the benchmark.
- **tilt**: Energy and Utilities stocks held at no more than half their benchmark weight.
- **forced turnover**: the trading that changes in index membership require, selling a stock that leaves and buying one that enters.
- **chosen turnover**: all other trading, which the optimiser chooses and the turnover cap limits.
- **allocation**, **selection** and **interaction**: the three parts of the active return in the Brinson-Fachler attribution. Allocation comes from holding sectors at weights other than the benchmark's, selection from returns within a sector that differ from the benchmark's, and interaction from the two together.
