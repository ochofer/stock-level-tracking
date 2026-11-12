"""
The source repository's values at commit 5414f14, kept so that the copies import, and not this study's
parameters. covariance.py and rules.py in this folder read their constants by the source's names; this file
holds every name they reference and no other, each at its value in src/bp/constants.py at that commit (SHA-256
constants.SOURCE_CONSTANTS_SHA256 in this package's parent), with the source line beside it. No module outside
this folder reads it: the study's own modules pass every parameter from src/slt/constants.py. Where a name means
something else in this study (ACTIVE_WEIGHT_BOUND is the source's bound per industry; the bound per stock here
is STOCK_ACTIVE_BOUND), the study's constant has its own name. tests/test_source_copies.py checks the names
against the copies' text and, where a clone of the source repository sits beside this one, the values against
the source's file at the commit.
"""

# Read by covariance.py
DAILY_TO_MONTHLY_SCALE = 21                                # line 327
COV_SCALE_VR_HORIZON_DAYS = 21                             # line 381
EXT_ESTIMATION_WINDOW = 120                                # line 214
EXT_SAMPLE_START = "1969-07"                               # line 219
EXT_COV_DAILY_WINDOW_YEARS = 3                             # line 325
EXT_COV_DAILY_WINDOW_YEARS_SENSITIVITY = (1, 5)            # line 326
EXT_COV_EWMA_HALFLIFE_DAYS = 252                           # line 328
COV_ESTIMATORS = ("sample", "ledoit_wolf", "factor", "pca")   # line 338
PCA_N_COMPONENTS = 5                                       # line 342
PCA_N_COMPONENTS_SENSITIVITY = 2                           # line 365
COV_TEST_ACTIVE_DRAWS = 200                                # line 371
COV_TEST_SEED = 20260910                                   # line 372
ACTIVE_WEIGHT_BOUND = 0.02                                 # line 290
CAPW_WEIGHT_TIMING = "same_month"                          # line 257
ROBUST_VARIANT = ("robust", "all_five")                    # line 411
EWMA_VARIANT = ("sample", "daily_5y_ewma")                 # line 438

# Read by rules.py
DGU_RISK_AVERSION = 1.0                                    # line 52
DGU_GMINC_LOWER_BOUND_FRACTION = 0.5                       # line 55
DGU_CONSTRAINED_BUDGET_INSIDE = True                       # line 67

# Read by optimiser.py (copied for milestone 3); ACTIVE_WEIGHT_BOUND above is read by it too
TURNOVER_CAP_MONTHLY_ONE_WAY = 0.02                        # line 291
TILT_INDUSTRIES = ("Coal", "Oil", "Util")                  # line 292
TILT_MAX_SHARE_OF_BENCHMARK = 0.5                          # line 293
OPT_SOLVER = "CLARABEL"                                    # line 390
OPT_SOLVER_OPTIONS = {"tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-10}   # line 391
OPT_SOLVER_OPTIONS_ROBUST = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}   # line 392
OPT_TOL = 1e-6                                             # line 393
FO_E1_BIND_TOL_ANNUAL = 0.0001                             # line 612
FO_SOLVER_OPTIONS_LOOSE = {"tol_gap_abs": 1e-7, "tol_gap_rel": 1e-7, "tol_feas": 1e-7}   # line 706
