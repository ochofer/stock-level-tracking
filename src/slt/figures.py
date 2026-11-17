"""
The data of the four figures of notebook 12, the dashboard and the note, read from the committed results only.

Every function reads files in results/<vintage>/ (constants.RESULTS_DIR) and nothing else, so the figures can be drawn
on any machine from a clone of the repository, with no extraction. Each returns a table in the units the figures show:
tracking error and the terms of the attribution in basis points a year, turnover in per cent a year. Nothing here is
by month: the four figures read the summaries over the window and the tables by year.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

from . import constants as C

# The four estimators in the order of every exhibit, each with the slot of the house palette it keeps (G3).
ORDER = ("ledoit_wolf", "sample", "factor", "pca")
LABELS = {"ledoit_wolf": "Ledoit-Wolf", "sample": "Sample", "factor": "Factor", "pca": "Principal-component"}
PALETTE = ("#24599e", "#d9711c", "#12929f", "#c8423b")
COLOURS = dict(zip(ORDER, PALETTE))
CAP_LABELS = {0.005: "0.5", 0.01: "1", 0.02: "2", 0.04: "4", None: "none"}
BP = 1e4


def results_dir(repo=None) -> pathlib.Path:
    """The committed results folder, from the repository root (default: two levels above this package)."""
    root = pathlib.Path(repo) if repo is not None else pathlib.Path(__file__).resolve().parents[2]
    return root / C.RESULTS_DIR


def _label(est: str, cs: str = C.MANDATE, cap=C.MAIN_CAP) -> str:
    return C.PATH_LABELS[(est, cs, cap)]


def _paths(res) -> dict:
    return json.loads((pathlib.Path(res) / "paths_summary.json").read_text())["paths"]


def tracking_error_by_estimator(res) -> pd.DataFrame:
    """Figure 1: forecast and realised tracking error of the four estimators under the mandate at the main cap,
    with the bias statistic and the months it rests on."""
    p = _paths(res)
    rows = []
    for est in ORDER:
        s = p[_label(est)]
        rows.append({"estimator": est, "label": LABELS[est], "realised": BP * s["realised_te"],
                     "forecast": BP * s["forecast_te_mean"], "bias": s["bias"], "bias_months": s["bias_months"],
                     "months": s["months"]})
    return pd.DataFrame(rows)


def tracking_error_by_year(res, est: str = "ledoit_wolf") -> pd.DataFrame:
    """Figure 2: realised and forecast tracking error by calendar year of one estimator under the mandate at the
    main cap, with the months each year holds."""
    t = pd.read_csv(pathlib.Path(res) / "paths_by_year.csv")
    t = t[t["path"] == _label(est)].sort_values("year")
    return pd.DataFrame({"year": t["year"].to_numpy(), "months": t["months"].to_numpy(),
                         "realised": BP * t["realised_te"].to_numpy(), "forecast": BP * t["forecast_te_mean"].to_numpy()})


def frontier(res) -> pd.DataFrame:
    """Figure 3: for each estimator of the frontier, at each cap, chosen turnover (per cent a year) and the realised
    and forecast tracking error (basis points a year)."""
    p = _paths(res)
    rows = []
    for est in C.FRONTIER_ESTIMATORS:
        for cap in C.FRONTIER_CAPS:
            s = p[_label(est, C.MANDATE, cap)]
            rows.append({"estimator": est, "label": LABELS[est], "cap": CAP_LABELS[cap],
                         "chosen_turnover": 100 * s["turnover_chosen_per_year"], "total_turnover": 100 * s["turnover_total_per_year"],
                         "realised": BP * s["realised_te"], "forecast": BP * s["forecast_te_mean"]})
    return pd.DataFrame(rows)


def sector_terms(res, est: str = "ledoit_wolf") -> pd.DataFrame:
    """Figure 4: each sector's three Brinson-Fachler terms over the window on one path under the mandate, as annual
    averages with their Newey-West standard errors, in the order of the GICS codes with Unclassified last."""
    b = json.loads((pathlib.Path(res) / "brinson_summary.json").read_text())["paths"][_label(est)]["sectors"]
    rows = []
    for code in C.ATTRIBUTION_SECTORS:
        s = b[code]
        row = {"sector": code, "label": C.GICS_SECTORS.get(code, C.UNCLASSIFIED)}
        for term in C.ATTRIBUTION_TERMS:
            row[term] = BP * s[f"{term}_per_year"]
            row[f"{term}_se"] = BP * s[f"{term}_per_year_se"]
        rows.append(row)
    return pd.DataFrame(rows)


def terms_beyond_two_se(res) -> dict:
    """For each of the four paths, the count of sector terms over the window that lie two Newey-West standard errors
    or more from zero, out of the twelve sectors' three terms, for the note under Figure 4."""
    paths = json.loads((pathlib.Path(res) / "brinson_summary.json").read_text())["paths"]
    out = {}
    for est in ORDER:
        sectors = paths[_label(est)]["sectors"]
        vals = [(s[f"{t}_per_year"], s[f"{t}_per_year_se"]) for s in sectors.values() for t in C.ATTRIBUTION_TERMS]
        out[est] = {"terms": len(vals), "beyond": sum(1 for v, se in vals if se > 0 and abs(v / se) >= 2)}
    return out


# ---------------------------------------------------------------------------
# Drawing: the static figures for the README and the note, in the house frame
# ---------------------------------------------------------------------------

INK, INK2, META, FRAME, GRID = "#262b33", "#5d6470", "#6b7280", "#cfcfcf", "#e5e5e5"
WIDTH = 6.85   # inches: the A4 text width of 174 millimetres


def style():
    """The house frame (G5) for matplotlib, with fixed element ids so that the same tables give the same file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path", "figure.dpi": 100,
                         "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": FRAME,
                         "xtick.color": FRAME, "ytick.color": FRAME, "xtick.labelcolor": META,
                         "ytick.labelcolor": META, "svg.hashsalt": "stock-level-tracking"})
    return plt


def save(fig, path):
    """Writes an SVG with no date and no creator, so that a rebuild from the same tables is byte-identical."""
    fig.savefig(path, format="svg", metadata={"Creator": None, "Date": None})


def draw_tracking_error(t: pd.DataFrame):
    """Figure 1: one row per estimator, the forecast hollow and the realised filled, joined, with the bias."""
    plt = style()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    for i, r in enumerate(t.itertuples()):
        c = COLOURS[r.estimator]
        ax.plot([r.forecast, r.realised], [i, i], color=c, lw=1.6, zorder=1)
        ax.plot(r.forecast, i, "o", mfc="white", mec=c, mew=1.6, ms=7, zorder=2)
        ax.plot(r.realised, i, "o", color=c, ms=7, zorder=2)
        ax.annotate(f"{r.forecast:.1f}", (r.forecast, i), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK2)
        ax.annotate(f"{r.realised:.1f}   bias {r.bias:.2f}", (r.realised, i), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["label"])
    ax.invert_yaxis()
    ax.set_xlim(30, 105)
    ax.set_ylim(len(t) - 0.4, -0.6)
    ax.set_xlabel("tracking error, basis points a year (hollow: mean forecast, filled: realised)")
    ax.grid(axis="x", lw=0.6, color=GRID)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def draw_by_year(t: pd.DataFrame, est: str = "ledoit_wolf"):
    """Figure 2: realised and mean forecast tracking error by year, one estimator."""
    plt = style()
    c = COLOURS[est]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    ax.plot(t["year"], t["realised"], "-o", color=c, lw=1.6, ms=4, label="realised")
    ax.plot(t["year"], t["forecast"], "--o", color=c, mfc="white", mec=c, lw=1.2, ms=4, label="mean forecast")
    peak = t.loc[t["realised"].idxmax()]
    ax.annotate(f"{peak.realised:.0f}", (peak.year, peak.realised), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.annotate(f"{peak.forecast:.0f}", (peak.year, peak.forecast), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK2)
    last = t.iloc[-1]
    ax.annotate(f"{last.realised:.0f}", (last.year, last.realised), xytext=(6, 4), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.annotate(f"{last.forecast:.0f}", (last.year, last.forecast), xytext=(6, -6), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_xlim(t["year"].min() - 0.5, t["year"].max() + 1.5)
    ax.set_xticks([y for y in range(2000, int(t["year"].max()) + 1, 4)])
    ax.set_ylim(0, 190)
    ax.set_ylabel("basis points a year")
    ax.grid(axis="y", lw=0.6, color=GRID)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig


def cap_labels(g: pd.DataFrame, gap: float = 2.0) -> list:
    """The cap labels of one estimator's realised dots, two dots closer than gap per cent of turnover a year sharing
    one label at their middle ("2, 4, none"), so that no two labels overlap."""
    out, run = [], []
    for r in g.sort_values("chosen_turnover").itertuples():
        if run and r.chosen_turnover - run[-1].chosen_turnover >= gap:
            out.append(run)
            run = []
        run.append(r)
    out.append(run)
    return [(sum(r.chosen_turnover for r in grp) / len(grp), max(r.realised for r in grp), ", ".join(r.cap for r in grp)) for grp in out]


def draw_frontier(t: pd.DataFrame):
    """Figure 3: realised (filled) and forecast (hollow) tracking error against chosen turnover, one dot per cap."""
    plt = style()
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    for est, g in t.groupby("estimator", sort=False):
        c = COLOURS[est]
        ax.plot(g["chosen_turnover"], g["realised"], "-o", color=c, lw=1.6, ms=5, label=f"{LABELS[est]}, realised")
        ax.plot(g["chosen_turnover"], g["forecast"], "--o", color=c, mfc="white", mec=c, lw=1.2, ms=5, label=f"{LABELS[est]}, mean forecast")
        for x, y, text in cap_labels(g):
            below = est == C.FRONTIER_ESTIMATORS[0]   # the first estimator's labels under its dots, the second's above
            ax.annotate(text, (x, y), xytext=(0, -13 if below else 7), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 65)
    ax.set_ylim(30, 95)
    ax.set_xlabel("chosen turnover, per cent a year (labels: the cap, per cent a month)")
    ax.set_ylabel("basis points a year")
    ax.grid(axis="y", lw=0.6, color=GRID)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout()
    return fig


def draw_sector_terms(t: pd.DataFrame, est: str = "ledoit_wolf"):
    """Figure 4: allocation and selection by sector over the window, each with a bar of two Newey-West standard
    errors on each side."""
    plt = style()
    c = COLOURS[est]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), sharey=True)
    y = range(len(t))
    for ax, term, title in zip(axes, ("allocation", "selection"), ("Allocation", "Selection")):
        v, se = t[term], t[f"{term}_se"]
        ax.errorbar(v, y, xerr=2 * se, fmt="o", color=c, ms=4.5, elinewidth=1.2, capsize=0)
        ax.axvline(0, color=FRAME, lw=0.8, zorder=0)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK, fontweight="bold")
        ax.set_xlim(-30, 30)
        ax.set_xlabel("basis points a year")
        ax.grid(axis="x", lw=0.6, color=GRID)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(t["label"])
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig
