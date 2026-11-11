"""
Pre-committed numbers for the study.

Every gate, expectation, sample boundary, mandate level and design parameter is
defined here, one per line, with its source beside it. A notebook or script may
read these names; it may not introduce numbers of its own. The design was fixed
on 9 November 2026, and the readings marked 10 November 2026 were fixed the
next day, all before any code that uses them was written. The amendments marked
11 November 2026 follow two findings in the extraction's structure, made from
counts alone and before any notebook judged a gate. A level changed after
a run that judges it carries the date and the reason beside it, so that a reader
can tell a specification from a number set after seeing a result.

Gates (G1 to G4) and expectations (E1 to E9) are strict: each is reported as met
or not met, in the same words either way. A gate not met stops the study, and
the data layer is published alone with the reason. An expectation not met is a
finding and is reported as one.

Sources cited by short name below:
  CRSP and Compustat: the extraction of 28 September 2026 from Wharton Research Data Services,
    accessed through Queen Mary University of London (section 1).
  French: Ken French's data library, downloaded at run time and never committed.
  The source repository: github.com/ochofer/optimal-vs-naive-diversification at
    commit 5414f14, whose constants are cited here by their names there.
  Brinson and Fachler (1985), "Measuring Non-U.S. Equity Portfolio Performance",
    Journal of Portfolio Management 11(3), 73-76.
  Shumway (1997), "The Delisting Bias in CRSP Data", Journal of Finance 52(1),
    327-340.
  Ledoit and Wolf (2004), "A well-conditioned estimator for large-dimensional
    covariance matrices", Journal of Multivariate Analysis 88(2), 365-411.
"""

# ---------------------------------------------------------------------------
# 1. The data: the extraction, read in place and never committed
# ---------------------------------------------------------------------------

# The folder that holds the extraction's manifest.json and raw/ is named by this
# environment variable, set on the machine that holds the data and outside the
# repository. Every notebook stops with one sentence when it is missing.
DATA_ROOT_ENV = "SLT_DATA_ROOT"

EXTRACTION_DATE = "2026-09-28"   # fetch date of every table used except the GICS history; the vintage of every result
EXTRACTION_SOURCE = ("CRSP and Compustat, via Wharton Research Data Services, "
                     "accessed through Queen Mary University of London")   # named with the extraction date on every exhibit drawn from them
MANIFEST_FILE = "manifest.json"  # the extraction's record: the SQL, rows, columns, SHA-256 and fetch time of every file
# The manifest the results are built from, and the one before it. The extraction
# module stops when the file on disk has another hash. The GICS history (section 4)
# was added on 10 November 2026 at 16:05 Amsterdam time, which changed the hash.
MANIFEST_SHA256, MANIFEST_SHA256_2026_09_28 = "23e96d189f4b095936f47a1ebacb4e987fb260726afab4b6e6400c25ebce8693", "8b2e6b6bf762204bb2c4f70b1a53c66d527d02d2905a540caa53275a54de2b22"   # 310 files and 112,161,796 rows from 10 November 2026; 309 files and 112,115,800 rows on 28 September 2026
CRSP_END = "2024-12-31"          # the last date of the CRSP library on this subscription; the sample ends here
DAILY_FILE_START = "1990-01-02"  # the first date of the daily stock file in the extraction

# The extraction's keys for the tables used (its folders raw/<key>/), with the WRDS table each holds.
TABLES = {
    "crsp_dsp500list": "crsp.dsp500list",       # S&P 500 membership by PERMNO with start and end dates: the universe
    "crsp_msp500": "crsp.msp500",               # CRSP's monthly file of the same universe: returns, count and total value
    "crsp_msf": "crsp.msf",                     # monthly prices, returns and shares outstanding
    "crsp_msedelist": "crsp.msedelist",         # delisting dates, codes and returns
    "crsp_msenames": "crsp.msenames",           # share codes through time
    "crsp_dsf": "crsp.dsf",                     # daily prices and returns, one file per year from 1990
    "crsp_ccm_lnkhist": "crsp.ccmxpf_lnkhist",  # the CRSP-Compustat link history
    "comp_idxcst_his": "comp.idxcst_his",       # Compustat's index constituent history, the cross-check on membership
    "comp_co_hgic": "comp.co_hgic",             # Compustat's GICS history by company (section 4)
    "comp_company": "comp.company",             # Compustat's current GICS, read only if the GICS history is unavailable
}
TABLES_NOT_USED = {
    "crsp_msp500list": "the same file as crsp_dsp500list (identical SHA-256 in the manifest), so it is read once",
}

# ---------------------------------------------------------------------------
# 2. The universe: the point-in-time S&P 500 from CRSP's own list
# ---------------------------------------------------------------------------

# A stock is in the universe in month t if its membership interval covers the
# last trading day of month t-1, so that nothing unknown at the start of the
# month decides membership. The last trading day of each month is CRSP's date
# for that month in its monthly index file.
MEMBERSHIP_RULE = "membership interval covers the last trading day of month t-1"
USABLE_COUNT_WITHIN = 10         # the first usable month is the first in which the universe's count is within 10 names of CRSP's count
SP500_GVKEYX = "000003"          # Compustat's identifier of the S&P 500, selecting the cross-check's rows of its constituent history
# Amended 11 November 2026. On this subscription Compustat's constituent history
# lists the members of September 2026 alone: 503 rows for the index, none with an
# end date. So the cross-check covers names still in the index and says nothing
# about removals. It is read on what the table holds, by year, with no verdict:
# (a) of the member-months of Compustat's members, each from the month after it
# joined to December 2024, the share found in the universe through the link, with
# the counts of rows, companies, rows that do not link and rows that joined after
# CRSP's last date; (b) for the same members, whether the two lists date the join
# to the same month, to within one month or further apart, CRSP's date being the
# start of the membership interval that covers Compustat's months; (c) the
# definition of 9 November, kept beside them and named for what it is: it counts
# every name that left the index before September 2026, which the table omits.
CROSS_CHECK_REPORT, CROSS_CHECK_REPORT_2026_11_09 = (
    "(a) share of Compustat's member-months found in the universe; (b) join dates on both lists; "
    "(c) the two-sided share, which counts the exits Compustat's table omits; by year, no verdict",
    "share of universe-months on which CRSP's and Compustat's lists disagree, by year; reported and not reconciled")   # the second superseded on 11 November 2026
CROSS_CHECK_JOIN_WITHIN_MONTHS = 1   # (b): joins dated to the same month, within one month, or further apart
SHARE_CODES_COMMON = (10, 11)    # share codes are not filtered, since the list is the universe; members outside these two are counted by year

# ---------------------------------------------------------------------------
# 3. Returns, delistings and market capitalisation
# ---------------------------------------------------------------------------

# Delisting returns, after Shumway (1997). CRSP's delisting return is applied in
# the month of the delisting where CRSP gives one. Where it is missing, the
# return is set by the delisting code, and every such month is counted.
DELIST_PERFORMANCE_CODES = (400, 599)   # performance-related delisting codes, inclusive
DELIST_PERFORMANCE_MISSING_RETURN = -0.30   # a missing delisting return with a performance-related code
DELIST_OTHER_MISSING_RETURN = 0.0           # a missing delisting return with any other code

# Fixed 10 November 2026: a universe member with no monthly return and no
# delisting event in the month is left out of that month's benchmark return and
# the other weights are rescaled to sum to one. The count is reported beside E1;
# when it is zero the rule is never used and the notebook says so.
MISSING_RETURN_RULE = "left out of the month's return, other weights rescaled to sum to one, counted"

# Market capitalisation at the start of month t is the absolute price times
# shares outstanding at the end of month t-1, in thousands of dollars as CRSP
# gives shares. A negative price in CRSP is the midpoint of bid and ask, so its
# absolute value is taken.
CAP_RULE = "|price| x shares outstanding at the end of month t-1, in thousands of dollars"
# A member's benchmark weight is its start-of-month capitalisation over the
# universe's sum, so the weight uses nothing unknown when the month begins, as
# CAPW_WEIGHT_TIMING does in the source repository.
BENCHMARK_WEIGHT_TIMING = "start of month"
# Fixed 11 November 2026: a member without a start-of-month capitalisation (no
# price or no shares at the end of month t-1, or a price of zero) has no weight in
# the month. It is left out of the benchmark return and the sector weights, the
# other weights are rescaled, and it is counted. The same exclusion holds for the
# optimiser's universe in that month. A negative price marks a bid and ask
# average: it is used in absolute value and is not counted as missing.
MISSING_CAP_RULE = "no weight in the month: out of the benchmark, the sector weights and the optimiser's universe, other weights rescaled, counted"

# ---------------------------------------------------------------------------
# 4. Sectors, through the CRSP-Compustat link
# ---------------------------------------------------------------------------

# PERMNO is the key; GVKEY attaches through the link history, with the link's
# dates inclusive. Fixed 10 November 2026, now for the Compustat cross-check: primary
# links only, and where a PERMNO-month still holds more than one GVKEY after these
# filters, the count is reported and the link with primary flag P is taken over C,
# then the one with the later start. The sector join follows the amendment below.
LINK_TYPES = ("LC", "LU")
LINK_PRIMARY_FLAGS = ("P", "C")  # the Compustat cross-check's links (section 2), as built and labelled
LINK_DATES_INCLUSIVE = True
LINK_TIE_BREAK = ("primary flag P before C", "then the later link start date")
# Amended 11 November 2026, for the sector join only: a member takes the GVKEY of
# the link of type LC or LU in force on the day, whatever its primary flag, in the
# order P, C, J, N, then the later link start where two remain. The member-months
# classified through a J or an N link are counted by year. Reason: the primary flag
# says which of a company's securities Compustat treats as primary, which matters
# going from a company to a security and not from a security to its company, and a
# second share class has its company's sector. The rule of 10 November, primary
# links only, is kept beside it as superseded.
SECTOR_LINK_FLAGS, SECTOR_LINK_FLAGS_2026_11_10 = ("P", "C", "J", "N"), ("P", "C")
SECTOR_LINK_TIE_BREAK = ("primary flag in the order P, C, J, N", "then the later link start date")

# The GICS history is one pull from Compustat added to the extraction on
# 10 November 2026, recorded in the manifest like every other file, because the
# extraction held only the current classification. Had it failed, the fallback
# was the current classification, disclosed as sectors as of the extraction
# date. The pull returned 45,996 rows, so the fallback is not used.
GICS_SOURCE = "comp_co_hgic"
GICS_FALLBACK = "comp_company"
# Fixed 10 November 2026: the sector of month t is the classification in force on
# the last trading day of month t-1, the day membership and capitalisation are read.
GICS_TIMING = "classification in force on the last trading day of month t-1"
GICS_OVERLAP_RULE = "where two classifications are in force on one day, the later start, counted"   # fixed 11 November 2026
GICS_LEVELS = ("gsector", "ggroup", "gind", "gsubind")   # sector, industry group, industry, sub-industry
# The eleven GICS sectors by code, under their current names. Real Estate (60) was
# separated from Financials (40) in 2016, and Telecommunication Services became
# Communication Services (50) in 2018; a month carries the code in force then.
GICS_SECTORS = {
    "10": "Energy", "15": "Materials", "20": "Industrials", "25": "Consumer Discretionary",
    "30": "Consumer Staples", "35": "Health Care", "40": "Financials", "45": "Information Technology",
    "50": "Communication Services", "55": "Utilities", "60": "Real Estate",
}
UNCLASSIFIED = "Unclassified"    # a universe-month with no sector: counted by year, and carried through the mandate and the attribution as a twelfth sector so that every identity holds
SECTOR_COUNT = len(GICS_SECTORS) + 1   # 12

# ---------------------------------------------------------------------------
# 5. Gates: a gate not met stops the study
# ---------------------------------------------------------------------------

G1_TE_MAX = 0.0050               # G1: the reconstruction's tracking error against CRSP's value-weighted return with dividends at most 50 basis points a year over the usable window
G2_USABLE_YEARS_MIN = 15         # G2: the usable window at least 15 years
G3_TRADING_DAYS = 10             # G3: a removed name has a CRSP price within 10 trading days of its removal date
G3_SHARE_MIN = 0.90              # G3: for at least 90% of removed names
# Fixed 10 November 2026: G3 is judged on the daily file, on removals from its
# first date to CRSP's last, the span the study uses. Removals before 1990 are
# counted and reported beside it with no verdict, on the monthly file, as the
# share with a price at the end of the removal month or the month before. A
# membership that ends on CRSP's last date is a current member and not a removal.
G3_WINDOW = (DAILY_FILE_START, CRSP_END)
G3_BEFORE_WINDOW = "share with a price at the end of the removal month or the month before, monthly file, no verdict"
G4_MONTH_SHARE_MIN = 0.90        # G4: market capitalisation present in at least 90% of a name's membership months
G4_NAME_SHARE_MIN = 0.90         # G4: for at least 90% of names
G4_WINDOW = "usable window, each name on its own membership months inside it"   # fixed 11 November 2026

# ---------------------------------------------------------------------------
# 6. Expectations: an expectation not met is a finding
# ---------------------------------------------------------------------------

# E1, the reconstruction. Over the usable window the benchmark's tracking error
# against CRSP's value-weighted return with dividends is at most 20 basis points
# a year and the mean monthly difference within 2 basis points of zero; the
# price-only benchmark against CRSP's price-only return meets the same two.
# Reason: CRSP computes its return from the same list and the same stock file,
# daily. The differences are mid-month membership changes, held here to the
# month's start, and CRSP's treatment of the removal day, which touch a small
# number of names a year at small weights. G1 at 50 is the cut; E1 at 20 says
# whether the layer reproduces the source closely enough to be called a
# reconstruction. The price-only benchmark is also shown against the index's own
# price return with no verdict: the index has weighted by float-adjusted shares
# since 2005, which CRSP's shares outstanding are not.
E1_TE_MAX = 0.0020
E1_MEAN_DIFF_MAX = 0.0002        # per month
# Fixed 11 November 2026: the price-only benchmark against the index's own price
# return over the months to December 2004, as tracking error and mean difference
# beside the whole-window figures, with no verdict. Until 2005 the index weighted
# by full capitalisation, so this gap is evidence on membership independent of
# CRSP's list, removals included, which Compustat's history cannot test.
INDEX_FULL_CAP_LAST_MONTH = "2004-12"

# E2, counts and values. In every month of the usable window the universe's count
# is within 10 of CRSP's count, and the summed capitalisation is within 2% of
# CRSP's total value in at least 95% of months. Reason: both are CRSP's own sums
# over the same list, so a gap can only be timing or a missing price, both of
# which the layer reports. Fixed 10 November 2026: CRSP measures its count and
# total value on the trading date of its row, so the universe at the start of
# month t is compared with CRSP's row for month t-1, the same day, in the same
# unit, thousands of dollars.
E2_COUNT_WITHIN = 10
E2_VALUE_REL_MAX = 0.02
E2_VALUE_MONTH_SHARE_MIN = 0.95
E2_ALIGNMENT = "universe at the start of month t against CRSP's row for month t-1"

# E3, French's answer key. Within the universe each month, value-weighted sorts on
# momentum and on variance; for each sort the universe's top-minus-bottom quintile
# spread against French's top-minus-bottom spread within his largest size
# quintile: correlation at least 0.80, and mean difference within two standard
# errors of the difference, over the months both exist. Reason: French's
# large-cap quintile holds more names than the universe and uses NYSE
# breakpoints, so a correlation near one is not expected; below 0.80 the
# universe is not the large-cap universe it claims to be.
E3_CORRELATION_MIN = 0.80
E3_MEAN_DIFF_SE = 2.0
SORT_QUANTILES = 5               # quintiles, because French's 25 portfolios are formed in quintiles
SORT_QUANTILES_REPORTED = 10     # deciles beside them, with no verdict
MOMENTUM_MONTHS = (12, 2)        # the return from month t-12 to month t-2, French's definition
VARIANCE_DAYS = 60               # the variance of daily returns over the previous 60 trading days, French's definition
VARIANCE_SORT_START = "1990-04"  # the first month with 60 daily returns before it

# E4, conditioning. In every month of the evaluation window the condition number
# of the sample covariance on daily data is at least 10 times that of the
# Ledoit-Wolf estimator, and under the mandate the sample estimator's realised
# tracking error is above the Ledoit-Wolf estimator's. Reason: about 500 names
# on about 756 days leave the smallest sample eigenvalues near zero, and
# shrinkage lifts them; the second clause says whether this shows in tracking.
E4_CONDITION_RATIO_MIN = 10.0

# E5, the frontier. For each of the two frontier estimators, the mean forecast
# tracking error does not rise from one cap to the next looser one across the
# five caps, and the realised tracking error at the 4% cap is below the realised
# tracking error at the 0.5% cap. Reason: a looser cap can only widen the
# feasible set, so the forecast cannot rise; the realised figure is noisy, so
# only the ends are compared.
E5_REALISED_CAPS_COMPARED = (0.04, 0.005)   # (looser, tighter)

# E6, the level. Under the mandate at the 2% cap, the realised tracking error of
# each of the three structured estimators lies between 50 and 150 basis points a
# year, and each one's bias statistic lies within 0.8 to 1.25. Reason: the same
# mandate on 49 industries realised 84.7 to 95.8 basis points a year in the
# source repository under the same tilt and bounds of the same order; the bias
# band is that repository's EVAL_BIAS_RANGE.
E6_TE_RANGE = (0.0050, 0.0150)
E6_BIAS_RANGE = (0.8, 1.25)
STRUCTURED_ESTIMATORS = ("ledoit_wolf", "factor", "pca")

# E7, forced turnover. Under the mandate at the 2% cap on the Ledoit-Wolf
# estimator, forced turnover is less than a third of total one-way turnover over
# the window. Reason: the index replaces about four to five per cent of its
# weight a year, while a cap of 2% a month allows about 24% a year of chosen
# turnover, and the source repository found the cap binding in nine months of ten.
E7_FORCED_SHARE_MAX = 1.0 / 3.0

# E8, the attribution. The three Brinson-Fachler terms sum to the active return
# within 1e-9 in every month and path; under the mandate on each structured
# estimator, the mean absolute monthly allocation effect exceeds the mean
# absolute monthly selection effect, and the mean absolute interaction effect is
# the smallest of the three. Reason: the identity is arithmetic; the mandate's
# active return comes from the tilt, a sector underweight, while the positions
# within sectors are chosen to minimise tracking error.
E8_IDENTITY_MAX_ABS_ERROR = 1e-9
E8_ORDER = ("allocation", "selection", "interaction")   # largest to smallest mean absolute effect

# E9, the factor cross-check. The source repository's factor attribution on the
# same active returns: contributions plus residual equal the active return
# within 1e-9 in every month, and the mean active exposure to the market factor
# is within 0.05 of zero for every mandate path. Reason: the source repository's
# ATTRIBUTION_MARKET_EXPOSURE_MAX, which beta neutrality to the benchmark left at
# 0.012 to 0.015 there; the same construction should leave the same order here.
E9_IDENTITY_MAX_ABS_ERROR = 1e-9
E9_MARKET_EXPOSURE_MAX = 0.05

GATES = ("G1", "G2", "G3", "G4")
EXPECTATIONS = ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9")

# ---------------------------------------------------------------------------
# 7. The study: the mandate, the paths and the measures
# ---------------------------------------------------------------------------

# Evaluation window: monthly rebalancing at the start of each month, with three
# years of daily returns before the first estimate, and CRSP's last month after
# the last. Amended 11 November 2026: Compustat's GICS history begins on 30 June
# 1999, so before July 1999 every member is Unclassified, and the sector bands,
# the tilt and the exclusions could not bind. The window starts in July 1999, with
# the first covariance estimate on daily returns from July 1996. The window of
# 9 November is kept beside it, superseded. 24 paths over 306 months are 7,344
# optimisation problems. E4 to E9 are judged on this window. The data layer, its
# usable window, G1 to G4 and E1 to E3 stay over the whole usable window.
EVAL_START, EVAL_START_2026_11_09 = "1999-07", "1993-01"
EVAL_END = "2024-12"
EVAL_MONTHS, EVAL_MONTHS_2026_11_09 = 306, 384
# Fixed 11 November 2026, before the coverage was seen: in the window's first
# month and in every later month, at least 95% of the universe's members and at
# least 95% of its capitalisation have a GICS sector. If July 1999 misses it, the
# window starts in the first month of 1999 that meets it, recorded with the
# figures. If no month of 1999 meets it, or a later month falls below, the study
# stops for a ruling.
EVAL_GICS_MEMBER_SHARE_MIN = 0.95
EVAL_GICS_CAP_SHARE_MIN = 0.95

# Covariance estimators, the source repository's four, on the previous three
# years of daily excess returns of the current universe members, scaled to
# monthly (EXT_COV_DAILY_WINDOW_YEARS, DAILY_TO_MONTHLY_SCALE, COV_ESTIMATORS,
# LW_TARGET, FACTOR_MODEL_FACTORS, PCA_FIRST_COMPONENT there).
COV_DAILY_WINDOW_YEARS = 3
DAILY_TO_MONTHLY_SCALE = 21      # 252 trading days a year divided by 12
COV_ESTIMATORS = ("sample", "ledoit_wolf", "factor", "pca")
LW_TARGET = "scaled_identity"    # Ledoit and Wolf (2004)
FACTOR_MODEL_FACTORS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")   # French's five factors plus momentum: B F B' plus a diagonal
PCA_FIRST_COMPONENT = "market"   # the benchmark's excess return is the first factor
PCA_FURTHER_COMPONENTS = 4       # four further components plus a diagonal
COV_MIN_DAILY_RETURNS = 252      # a member with fewer daily returns in the window takes the factor model's estimate of its row in every estimator; the count is reported by month
SAMPLE_MONTHLY_STATUS = "unavailable: about 500 names on 120 months give a singular matrix"

# The mandate: the source repository's constraint set C3 at the stock level,
# cumulative in four sets reported in this order.
LONG_ONLY = True
STOCK_ACTIVE_BOUND = 0.0050      # each stock within 50 basis points of its benchmark weight, in absolute terms; the source repository's 2-percentage-point bound per industry does not fit stock weights that run from a basis point to 7%
STOCK_ACTIVE_BOUND_SENSITIVITY = 0.0100   # the one sensitivity, reported with no verdict
SECTOR_BAND = 0.02               # each of the twelve sectors, Unclassified included, within 2 percentage points of its benchmark weight: the source repository's ACTIVE_WEIGHT_BOUND at the sector level
TURNOVER_CAP_MONTHLY_ONE_WAY = 0.02   # chosen one-way turnover at most 2% a month, against drifted weights; forced turnover is outside the cap and reported separately
TILT_SECTORS = ("10", "55")      # Energy and Utilities, the stock-level counterpart of Coal, Oil and Utilities in the source repository
TILT_MAX_SHARE_OF_BENCHMARK = 0.5     # each stock in a tilt sector held at no more than half its benchmark weight
# Weight zero for two sub-industries, the counterparts of French's Smoke and Guns.
# A sub-industry is wider than a list of controversial weapons, and every text
# that names the exclusion says so.
EXCLUDED_SUBINDUSTRIES = {"30203010": "Tobacco", "20101010": "Aerospace & Defense"}
BETA_NEUTRAL_RULE = "beta to the benchmark from the estimator's own covariance over the same window"
CONSTRAINT_SETS = {
    "C0": ("long_only",),
    "C1": ("long_only", "stock_bound", "sector_band"),
    "C2": ("long_only", "stock_bound", "sector_band", "turnover_cap"),
    "C3": ("long_only", "stock_bound", "sector_band", "turnover_cap", "tilt", "exclusion", "beta_neutral"),
}
MANDATE = "C3"
# The infeasible-month rule of the source repository (OPT_D5_MONTH there): the
# optimiser finds the smallest feasible turnover first and never poses a capped
# problem with no solution; every returned portfolio is checked against every
# constraint at OPT_TOL.
INFEASIBLE_MONTH_RULE = "smallest feasible turnover first"
OPT_SOLVER = "CLARABEL"          # the source repository's solver, deterministic
OPT_SOLVER_OPTIONS = {"tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-10}   # the source repository's OPT_SOLVER_OPTIONS
OPT_TOL = 1e-6                   # feasibility checks on weights, bounds and turnover; "binding" means within this of the limit

# Paths. Every path starts as the benchmark. Sixteen are the four estimators
# under the four constraint sets, with the 2% cap where the set has a cap; eight
# are the Ledoit-Wolf and factor estimators under the mandate at four further
# caps, which with the 2% runs give the frontier at five caps. A cap of None
# means no turnover cap. Nothing is added after the first run without a dated
# disclosure here.
MAIN_CAP = TURNOVER_CAP_MONTHLY_ONE_WAY
FRONTIER_CAPS = (0.005, 0.01, 0.02, 0.04, None)
FRONTIER_ESTIMATORS = ("ledoit_wolf", "factor")
START_PORTFOLIO = "benchmark"
PATHS = tuple(
    [(est, cs, MAIN_CAP if "turnover_cap" in CONSTRAINT_SETS[cs] else None)
     for cs in CONSTRAINT_SETS for est in COV_ESTIMATORS]
    + [(est, MANDATE, cap) for est in FRONTIER_ESTIMATORS for cap in FRONTIER_CAPS if cap != MAIN_CAP]
)
PATH_COUNT = 24
SPEC_COUNT = PATH_COUNT          # the specification count, written down before the code runs and reported

# Measures, per path: forecast and realised tracking error (annualised), the bias
# statistic and its band, one-way turnover split into chosen and forced, the cost,
# gross and net active return with Newey-West standard errors, the share of
# months in which the cap binds, the largest cumulative fall of the active
# return, and the rolling 36-month tracking error.
COST_PER_UNIT_TURNOVER = 0.0010  # 10 basis points per unit of one-way turnover: liquid large caps
COST_SENSITIVITY = 0.0020        # the one cost sensitivity
NEWEY_WEST_LAGS = 12
BIAS_BAND = E6_BIAS_RANGE
ROLLING_TE_MONTHS = 36

# Turnover decomposition: one-way turnover each month in three parts that sum to
# the total. Membership is the weight of names leaving plus the benchmark weight
# of names entering; drift is what restoring last month's target weights after
# price moves would take, net of membership; rebalancing is the rest.
TURNOVER_PARTS = ("membership", "drift", "rebalancing")

# The Brinson-Fachler attribution (Brinson and Fachler 1985), monthly, by the
# twelve sectors, for the mandate's four paths at the 2% cap. With w and b the
# start-of-month portfolio and benchmark sector weights and R the sectors'
# returns in the month: allocation (w_s - b_s)(Rb_s - Rb), selection
# b_s(Rp_s - Rb_s), interaction (w_s - b_s)(Rp_s - Rb_s). Over the window the
# monthly terms are summed arithmetically and reported as annual averages with
# standard errors, with no geometric linking and no smoothing.
ATTRIBUTION_TERMS = E8_ORDER
ATTRIBUTION_PATHS = tuple((est, MANDATE, MAIN_CAP) for est in COV_ESTIMATORS)
ATTRIBUTION_LINKING = "arithmetic sum of monthly terms"
FACTOR_ATTRIBUTION_FACTORS = FACTOR_MODEL_FACTORS   # the cross-check, E9

# ---------------------------------------------------------------------------
# 8. Code copied from the source repository
# ---------------------------------------------------------------------------

# Copied verbatim at the commit below into src/slt/, with the source path and the
# commit in each copy's first line. The SHA-256 is that of the source file at the
# commit; a test hashes each copy without its first line and compares. Any change
# to a copy is recorded here with its date and reason.
SOURCE_REPOSITORY = "https://github.com/ochofer/optimal-vs-naive-diversification"
SOURCE_COMMIT = "5414f14"
COPIED_FILES = {
    "optimiser.py": ("src/bp/optimiser.py", "8944b967bb5969fe3c4b2ffbc603aa2b75a6fddc0683735fc842ec241e303b93"),
    "covariance.py": ("src/bp/covariance.py", "55cb25bb4aec83d7b1924248518a5b72a87b193263d383e2e93422779fe47534"),
    "attribution.py": ("src/bp/attribution.py", "63a5536c08a0ae25f90c5af954c05856c3ff6ef5b33ee7bf31545194ffeb9f1e"),
}
# Read at the same commit for the rolling evaluation and its measures, and adapted
# in this repository's own modules; hashed so that the version read is on record.
SOURCE_ADAPTED_FILES = {
    "evaluation.py": ("src/bp/evaluation.py", "34f0817368cb69e00eee076bdb5c7bccb859eb952a6b0ffbbc5812e21c5d5d0e"),
    "backtest.py": ("src/bp/backtest.py", "6bf73b011560cbf720744e88ec24805159d4e6d9ee6b3818c810e510e2aa14de"),
}

# ---------------------------------------------------------------------------
# 9. Outputs and what is committed
# ---------------------------------------------------------------------------

RESULTS_DIR = "results/20260928"  # named by the extraction date, the vintage
PROVENANCE_FILE = RESULTS_DIR + "/provenance.json"   # the extraction date, the manifest's hash, the fetch date of every table used, French's vintage line, the versions of the tools
# Fixed 10 November 2026. By month the repository holds only what the study
# computes itself. Of the comparisons with CRSP's index file it holds only summary
# figures, over the window and by year: never a CRSP series by month, and never a
# monthly difference from one, because the benchmark's return beside such a
# difference would rebuild the licensed series. No per-security row, no list of
# PERMNOs or GVKEYs with dates and no per-stock weight is ever committed.
# Amended 11 November 2026: the benchmark's return, the universe's count and its
# summed capitalisation by month reproduce CRSP's index file in their values, so
# none of the three is committed by month; the by-year tables and the summaries
# stay. By month the repository holds only the eleven GICS sector weights as shares
# of the classified capitalisation, July 1999 to December 2024. The list of
# 10 November is kept beside it as superseded.
COMMITTED_MONTHLY, COMMITTED_MONTHLY_2026_11_10 = (
    ("eleven GICS sector weights as shares of the classified capitalisation, July 1999 to December 2024",),
    ("benchmark return with dividends", "benchmark return, price-only", "universe count",
     "universe summed capitalisation", "benchmark sector weights"),
)
# Fixed 11 November 2026, for every table by month: a cell is committed only if it
# aggregates at least three members and no member is more than 80% of it. A sector
# with no member in a month has weight zero, holds no member's value and is
# committed as zero. Unclassified does not pass and is never committed by month.
# Notebook 05 checks the rule on what it writes and stops if a cell fails.
SMALL_CELL_MIN_MEMBERS = 3
SMALL_CELL_MAX_MEMBER_SHARE = 0.80
# Fixed 11 November 2026: where the data root is present, the reconstruction test
# also fails if a committed column by month equals a field of CRSP's index file to
# the sixth decimal in more than 5% of months.
RECONSTRUCTION_EQUAL_DECIMALS = 6
RECONSTRUCTION_EQUAL_MONTHS_MAX = 0.05
COMPARISON_SUMMARIES = ("tracking error", "mean difference", "correlation", "months outside the tolerance")
TIMEZONE = "Europe/Amsterdam"
SEED = 20261109                  # for the tests' made-up data; the design has no random draw
