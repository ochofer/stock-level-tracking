# Stock-level index tracking: a point-in-time benchmark, a long-only mandate, and its performance attribution by sector

This study is in build and at milestone 2 of 4, the answer key and the covariance estimators.

I build a point-in-time S&P 500 universe from CRSP and Compustat and check it in three ways: against CRSP's own return, count and total value for the same universe, and against Ken French's portfolios formed on size and momentum and on size and variance. On that universe I then run a long-only mandate that holds close to the benchmark, under four covariance estimators from July 1999 to December 2024, split its turnover into the part that changes in index membership force and the part the optimiser chooses, and attribute its active return by sector with the method of Brinson and Fachler (1985).

## The data layer

The universe in month t is the set of stocks in CRSP's list of index members on the last trading day of month t minus 1, so that nothing unknown when the month begins decides membership. Each member's monthly return comes from CRSP's monthly stock file. A delisting in the month is applied as Shumway (1997) proposes: CRSP's delisting return where it gives one, and where it does not, minus 30 per cent for a delisting for poor performance and zero for any other reason. A member's market capitalisation at the start of month t is its price times its shares outstanding at the end of month t minus 1, and the benchmark weights each member by it. A member's sector is the GICS sector in force on the last trading day of month t minus 1, found through CRSP's link to Compustat. A member with no sector on that day is counted in a twelfth sector, Unclassified. The index as S&P publishes it is not the benchmark: its total return is not in the extraction, and its float adjustment, which counts only the shares available to the public, cannot be rebuilt from CRSP's share counts. CRSP ends on 31 December 2024 on this subscription, and so does the sample.

Two further comparisons have no verdict. Compustat's history of the index's members is set beside CRSP's list, by year. On this subscription that history lists the members of September 2026 alone, so the cross-check covers names still in the index and says nothing about removals. The benchmark's price-only return is also set beside the index's own price return over the months to December 2004. Until 2005 the index weighted each stock by its full capitalisation, so that gap is evidence on membership independent of CRSP's list, removals included.

The evaluation window of the mandate runs from July 1999 to December 2024, 306 months, because Compustat's GICS history begins on 30 June 1999 and the mandate's sector constraints need a sector for every member. It holds only if in every one of those months at least 95 per cent of the universe's members and at least 95 per cent of its capitalisation have a sector, a condition fixed before the coverage was seen.

Notebooks 01 to 05 build each part from the extraction and run four gates and two expectations on it:

- **G1**: the benchmark's tracking error against CRSP's own value-weighted return of the same list is at most 50 basis points a year over the usable window.
- **G2**: the usable window is at least 15 years long.
- **G3**: at least 90 per cent of the stocks removed from the universe from 2 January 1990 to 31 December 2024 have a CRSP price within 10 trading days of their removal.
- **G4**: the market capitalisation is present in at least 90 per cent of membership months for at least 90 per cent of names.
- **E1**: the tracking error is at most 20 basis points a year and the mean monthly difference within 2 basis points of zero, with dividends and price-only.
- **E2**: the universe's count is within 10 names of CRSP's count in every month of the usable window, and its summed capitalisation within 2 per cent of CRSP's total value in at least 95 per cent of months.

## Results of the data layer

Notebooks 01 to 05 ran on 11 November 2026 on the extraction of 28 September 2026, with the GICS history added on 10 November 2026. Every figure below is in `results/20260928/`, and the six verdicts are in `verdicts_milestone1.json`. All four gates and both expectations are met.

- **The universe** holds 1,936 securities over 2,064 membership spells, 440,724 member-months from January 1926 to December 2024. Its count is within 10 names of CRSP's own count on the same day in all 1,188 months, so the usable window is the whole 99 years (G2).
- **The reconstruction.** The benchmark's tracking error against CRSP's value-weighted return of the same list is 11.5 basis points a year with dividends and 11.6 price-only, with mean monthly differences of minus 0.18 and minus 0.16 basis points (G1 and E1). Over the evaluation window, July 1999 to December 2024, the tracking error with dividends is 15.2 basis points a year, with no verdict. Its summed capitalisation is within 2 per cent of CRSP's total value in 1,186 of the 1,188 months (E2).
- **Removals and capitalisations.** All 836 removals from 2 January 1990 have a CRSP price within 10 trading days (G3). Of the 725 removals before 1990, 658 have a price at the end of the removal month or the month before, reported with no verdict. 1,902 of 1,912 names have a market capitalisation in at least 90 per cent of their membership months (G4).
- **Delistings.** Of the 799 delisting months of members, CRSP gives the delisting return in 798, and the minus 30 per cent of Shumway (1997) is used once.
- **The index's own price return**, with no verdict: a tracking error of 93.6 basis points a year over the months to December 2004 and 22.4 from January 2005.
- **Compustat's list.** Of the member-months of Compustat's 468 members that joined by December 2024, the universe holds at least 99.1 per cent in every year from 1990. Both lists date 339 of the 347 joins from 1990 to the same month.
- **Sectors.** A member takes its company's sector through any link of the two link types, so a second share class has its company's sector. From July 1999 to December 2024 at least 99.4 per cent of the universe's members and at least 99.0 per cent of its capitalisation have a GICS sector in every month, so the evaluation window runs the 306 months from July 1999. Every cell of the table of sector weights by month sums at least three securities, and none is more than 58.4 per cent of its cell.

## The answer key and the covariance estimators

Notebook 06 sorts the universe each month on two signals known at the start of the month and compares the result with Ken French's portfolios, which are built from the same CRSP files by other rules. Momentum is the return compounded from month t minus 12 to month t minus 2, and variance is the variance of daily returns over the 60 trading days before the month, both as French defines them and with his conditions for a stock to enter the sort. The members are cut into five groups of equal size, the quintiles, and the spread is the cap-weighted return of the highest quintile minus that of the lowest. French's spread is the same within his largest size quintile, the stocks above the 80th percentile of the market capitalisations of NYSE stocks. That quintile holds between about 100 and 530 names a month, fewer than the universe's 500 in almost every month from April 1957 and more than its 90 before, so the two sets overlap without coinciding.

Notebook 07 estimates, for each month of the evaluation window, the covariance matrix of the members' daily excess returns over the three years before the month in the four ways the mandate will use: the sample covariance, the Ledoit-Wolf estimator, which shrinks the sample covariance towards a scaled identity matrix (Ledoit and Wolf 2004), a factor model on French's five factors and his momentum factor, and a principal-component model with the benchmark as its first factor. The four are copied unchanged from the repository of Optimal versus naive diversification at commit 5414f14, and a test checks each copied file's SHA-256. A member that lacks a return on more than five days of the window, about 11 a month, takes a row from a model fitted on its own days, and each of the four matrices is built so that it is positive semi-definite by construction.

- **E3**: for the momentum sort and for the variance sort, the universe's spread has a correlation of at least 0.80 with French's, and their mean monthly difference is within two standard errors of zero.
- **E4, first clause**: in every month of the evaluation window, the condition number of the sample covariance matrix is at least 10 times that of the Ledoit-Wolf matrix. The second clause, on the mandate's tracking error, is judged in milestone 3.

## Results of milestone 2

Notebooks 06 and 07 ran on 12 November 2026 on the same extraction, with French's files built on CRSP's database of September 2026 and downloaded at run time. Every figure below is in `results/20260928/`, and the three verdicts are in `verdicts_milestone2.json`. E3 is met for both sorts. The first clause of E4 is not met.

- **Momentum** (E3 met). Over the 1,172 months from January 1927 to December 2024 with a value in both, the universe's spread has a correlation of 0.888 with French's, and their mean monthly difference is minus 10.4 basis points on a standard error of 9.0, or 1.15 standard errors. With no verdict, the correlation is 0.846 over the 359 months to March 1957, when the universe held about 90 names, and 0.927 over the 813 months from April 1957. With ten groups in place of five it is 0.875, and the mean difference plus 23.6 basis points on a standard error of 11.4.
- **Variance** (E3 met). Over the 417 months from April 1990, the correlation is 0.954 and the mean monthly difference plus 2.7 basis points on a standard error of 9.4, or 0.29 standard errors. With ten groups, with no verdict, 0.912 and plus 12.6 on 17.2.
- **The stocks left out of the sorts.** Of 439,656 member-months from January 1927, 2,539 have no price 13 months back, 103 have no return two months back and 152 have no capitalisation, and 548 are sorted with one month missing from the compounding. Of 209,020 member-months from April 1990, 76 have fewer than 20 daily returns and 9 have no capitalisation.
- **Conditioning** (E4's first clause not met). The sample matrix's condition number is at least 10 times the Ledoit-Wolf matrix's in 86 of the 306 months, all of them from 2017. The ratio is 5.4 at the median and 2.8 at the least, in August 2012. With about 490 members on about 755 days the sample matrix is not singular, and the Ledoit-Wolf shrinkage intensity lies between 1.1 and 7.1 per cent, so the shrunk matrix stays close to the sample one. The ratio is above 10 in every month from 2018 to 2022, and the sample matrix's condition number is highest in 2020, 381,000 at the median against 5,000 for the Ledoit-Wolf matrix. The months that reach the level have another cause. From 2014 the universe holds up to five companies with two share classes each. Where both classes of a company are complete members, their daily returns have a correlation of 0.98 at the median, so the difference between the two is a portfolio with almost no variance, and it sets the smallest eigenvalue of the sample matrix. With one share class for each company the level is reached in 30 of the 306 months, all from 2020 to 2022, and the median ratio is 5.0. This check was added after the run and carries no verdict.
- **Members without a complete window.** A month has 490 complete members on average and 478 at the least, and 11.4 members with a row from a model on average and 22 at the most, of whom 3.2 on average had fewer than 252 daily returns. 190 member-months in 134 months had fewer than 60 and took the mean loadings of their sector's complete members, or of the whole universe's. At most 9 days were left out of a window. Every matrix passed the eigenvalue check in every month.

## What is pre-committed

Every gate, expectation, sample boundary and mandate level is in `src/slt/constants.py`, one per line with its reason and the date it was fixed, before any code that judges it has run. A gate that is not met stops the study, and the data layer is then published alone with the reason. An expectation that is not met is reported as a finding, in the same words as one that is met.

## Data

The stock data are CRSP and Compustat, via Wharton Research Data Services, accessed through Queen Mary University of London, from an extraction of 28 September 2026. They are licensed, and this repository holds none of them: it holds code, tests, the SHA-256 of the extraction's manifest and aggregate results. Nothing that reproduces CRSP's index file is committed by month, so the benchmark's return, the universe's count and its summed capitalisation appear only in summaries and by year. The one table by month holds the eleven GICS sector weights as shares of the classified capitalisation, from July 1999 to December 2024. A figure by month is committed only if it sums at least three securities and no security is more than 80 per cent of it, so the Unclassified sector is never committed by month. Ken French's data library is downloaded at run time and is not committed either.

## How to run it

1. Obtain the extraction. Each file's query and SHA-256 are in `results/20260928/extraction_files.csv`. CRSP revises its history once a year, so the same queries run later return a later vintage, with other hashes.
2. Set the environment variable `SLT_DATA_ROOT` to the folder that holds the extraction's `manifest.json`, outside the repository. Notebook 01 stops with one sentence when the manifest's SHA-256 differs from `MANIFEST_SHA256` in `src/slt/constants.py`.
3. Install the requirements with `pip install -r requirements.txt` and run the notebooks in `notebooks/` in order, from 01. Each writes its aggregates to `results/20260928/`. Notebooks 06 and 07 download four files from Ken French's data library into `data/french/`, which git ignores, and record each file's SHA-256 in `provenance_french.json`.
4. Run `python -m pytest tests`. Beside the unit tests, the reconstruction test reads every file in `results/` and fails if one names a security, carries a series of CRSP's index file or holds more rows than an aggregate needs. On a machine with the extraction it also fails if a column by month equals a field of CRSP's index file, to the sixth decimal, in more than 5 per cent of months. The tests also check that the files copied from the repository of Optimal versus naive diversification are unchanged, and, where a clone of that repository sits beside this one, that the constants they read equal its own at commit 5414f14.

The notebooks are built by `build_notebooks.py` and committed without outputs.

## References

Brinson, G. P. and Fachler, N. (1985). Measuring non-U.S. equity portfolio performance. *Journal of Portfolio Management* 11(3), 73 to 76.

Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics* 116(1), 1 to 22.

Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2), 365 to 411.

Shumway, T. (1997). The delisting bias in CRSP data. *Journal of Finance* 52(1), 327 to 340.

## Terms

- **universe**: the point-in-time S&P 500, that is, the stocks in CRSP's list of index members on the last trading day of the month before.
- **benchmark**: the cap-weighted universe, in which each stock's weight is its market capitalisation at the start of the month divided by the universe's total.
- **data layer**: the universe with its returns, delisting returns, market capitalisations and sectors, together with the checks run on it before the study uses it.
- **usable window**: the months from the first month in which the universe's count is within 10 names of CRSP's own count, to December 2024, the last month of CRSP on this subscription.
- **delisting return**: the return from a stock's last price on its exchange to the value its holders received when it was delisted.
- **GICS**: the Global Industry Classification Standard of S&P and MSCI, with 11 sectors and, below them, industry groups, industries and sub-industries.
- **Unclassified**: the twelfth sector, holding every member-month with no GICS sector on the day membership is read.
- **quintile**: one of five groups of equal numbers of stocks, ranked on a signal. A **spread** is the cap-weighted return of the highest quintile minus that of the lowest.
- **excess return**: a return minus the risk-free rate, here French's one-month Treasury bill rate.
- **covariance matrix**: the table of the variances of the stocks' returns and of their covariances with each other.
- **eigenvalue**: the variance of one of the uncorrelated portfolios into which a covariance matrix can be split. A matrix is **singular** when its smallest eigenvalue is zero, and **positive semi-definite** when none is negative.
- **condition number**: a covariance matrix's largest eigenvalue divided by its smallest. A large one means small errors in the matrix become large errors in the weights an optimiser draws from it.
- **shrinkage intensity**: the weight, from zero to one, that the Ledoit-Wolf estimator puts on its scaled identity matrix.
- **complete member**: a member with a return on every day of the three-year window but at most five. Every other member takes a row from a model fitted on its own days.
- **gate**: a test of the data layer fixed in advance. A gate that is not met stops the study.
- **expectation**: a test of a result fixed in advance. An expectation that is not met is reported as a finding.
- **active return**: the portfolio's return minus the benchmark's in the same month.
- **tracking error**: the standard deviation of the monthly active return, times the square root of 12.
- **mandate**: the full set of constraints the portfolio obeys. It is long-only, holds each stock within 50 basis points of its benchmark weight and each sector within 2 percentage points, caps chosen turnover, applies the tilt, excludes two sub-industries and keeps a beta of one to the benchmark.
- **tilt**: Energy and Utilities stocks held at no more than half their benchmark weight.
- **forced turnover**: the trading that changes in index membership require, selling a stock that leaves and buying one that enters.
- **chosen turnover**: all other trading, which the optimiser chooses and the turnover cap limits.
- **allocation**, **selection** and **interaction**: the three parts of the active return in the Brinson-Fachler attribution. Allocation comes from holding sectors at weights other than the benchmark's, selection from returns within a sector that differ from the benchmark's, and interaction from the two together.

## Other projects

**Optimal versus naive diversification.** Under an index-tracking mandate on French's 49 industries, five covariance estimators understate their own tracking error, by 22 to 61 per cent. The replication of DeMiguel, Garlappi and Uppal (2009) reproduces 29 of the 30 Sharpe ratios that carry a band. [Page](https://www.carlohofer.com/projects/optimal-vs-naive-diversification/), [dashboard](https://www.carlohofer.com/optimal-vs-naive-diversification/dashboard/B-P_dashboard.html), [note](https://www.carlohofer.com/optimal-vs-naive-diversification/note/B-P_note_2026-10.pdf), [code](https://github.com/ochofer/optimal-vs-naive-diversification)

**Climate risk as an equity signal.** An ownership graph linking 328 listed companies to the 5,115 tracked assets they own. Of 19 apparent ownership changes checked by hand, 12 had no transaction behind them. [Page](https://www.carlohofer.com/projects/climate-risk-equity-signal/), [note](https://github.com/ochofer/paper1-hazard-exposure-data/releases/download/note-v1.2/what-a-vintage-difference-measures.pdf), [code](https://github.com/ochofer/paper1-hazard-exposure-data)

**Volatility-managed portfolios: a replication, the real-time critique, and the months after the sample.** Volatility estimation and position sizing on Ken French's equity factors, backtested out of sample and without look-ahead, replicating Moreira and Muir (2017) and Cederburg, O'Doherty, Wang and Yan (2020). In build: its findings are stated when the study is complete. [Page](https://www.carlohofer.com/projects/volatility-managed-portfolios/), [code](https://github.com/ochofer/volatility-managed-portfolios)

**Rules-based portfolio.** Runs a real-money portfolio under rules published before the first order: two UCITS ETFs at 70/30, a monthly cycle with pre-trade checks, look-through exposure and ex-ante risk on a public dashboard. [Page](https://www.carlohofer.com/projects/rules-based-portfolio/), [dashboard](https://www.carlohofer.com/rules-based-portfolio/), [guide](https://www.carlohofer.com/rules-based-portfolio/guide/guide.pdf), [code](https://github.com/ochofer/rules-based-portfolio)
