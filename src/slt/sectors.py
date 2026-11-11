"""
Sectors: GICS by month for every universe member, through the CRSP-Compustat link.

PERMNO is the key. A GVKEY attaches through the link history, primary links only, on the dates each link
was valid (constants.LINK_TYPES, LINK_PRIMARY_FLAGS, LINK_TIE_BREAK). The sector of month t is the GICS
classification in force on the last trading day of month t-1 (constants.GICS_TIMING). A member with no
GVKEY, or with no classification in force, is Unclassified, a twelfth sector carried through the mandate
and the attribution. Nothing here reads or writes a file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C

OPEN_END = pd.Timestamp("2099-12-31")   # stands in for a missing end date, which means the record is still in force


def _links(links: pd.DataFrame, flags) -> pd.DataFrame:
    keep = links["linktype"].isin(C.LINK_TYPES) & links["linkprim"].isin(flags) & links["lpermno"].notna()
    out = links.loc[keep, ["gvkey", "lpermno", "linkprim", "linkdt", "linkenddt"]].copy()
    out["permno"] = out["lpermno"].astype("int64")
    out["linkenddt"] = out["linkenddt"].fillna(OPEN_END)
    return out.drop(columns="lpermno")


def primary_links(links: pd.DataFrame) -> pd.DataFrame:
    """The links of the Compustat cross-check: the two link types, primary flags P and C only, with a PERMNO,
    end dates filled."""
    return _links(links, C.LINK_PRIMARY_FLAGS)


def sector_links(links: pd.DataFrame) -> pd.DataFrame:
    """The links of the sector join: the two link types with any primary flag (constants.SECTOR_LINK_FLAGS),
    with a PERMNO, end dates filled. A second share class reaches its company, and so its sector, through a
    link that is not flagged primary."""
    return _links(links, C.SECTOR_LINK_FLAGS)


def link_gvkeys(ulong: pd.DataFrame, links: pd.DataFrame) -> tuple:
    """The GVKEY of each universe member in each month, from the link valid on the last trading day of the
    month before, both ends of the link included, with the link's primary flag. Where more than one link is
    valid, the flags are taken in the order of constants.SECTOR_LINK_FLAGS (P, C, J, N), then the later
    start. Returns the frame and the count of member-months with more than one valid link."""
    m = ulong.merge(links, on="permno", how="inner")
    m = m[(m["linkdt"] <= m["date_prev"]) & (m["date_prev"] <= m["linkenddt"])].copy()
    multiple = int((m.groupby(["permno", "month"]).size() > 1).sum())
    order = {flag: i for i, flag in enumerate(C.SECTOR_LINK_FLAGS)}
    m["flag_rank"] = m["linkprim"].map(order).fillna(len(order))
    m = (m.sort_values(["permno", "month", "flag_rank", "linkdt"], ascending=[True, True, True, False])
          .drop_duplicates(["permno", "month"]))
    out = ulong.merge(m[["permno", "month", "gvkey", "linkprim"]], on=["permno", "month"], how="left")
    return out, multiple


def gics_by_month(linked: pd.DataFrame, history: pd.DataFrame) -> tuple:
    """The GICS sector, industry group, industry and sub-industry of each member in each month, from the
    classification in force on the last trading day of the month before. Returns the frame and the count
    of stock-months on which more than one classification was in force (the later start is then taken)."""
    h = history[["gvkey", "gsector", "ggroup", "gind", "gsubind", "indfrom", "indthru"]].copy()
    h["indthru"] = h["indthru"].fillna(OPEN_END)
    m = linked.dropna(subset=["gvkey"]).merge(h, on="gvkey", how="inner")
    m = m[(m["indfrom"] <= m["date_prev"]) & (m["date_prev"] <= m["indthru"])]
    overlapping = int((m.groupby(["permno", "month"]).size() > 1).sum())
    m = (m.sort_values(["permno", "month", "indfrom"], ascending=[True, True, False])
          .drop_duplicates(["permno", "month"]))
    out = linked.merge(m[["permno", "month", "gsector", "ggroup", "gind", "gsubind"]], on=["permno", "month"], how="left")
    out["sector"] = out["gsector"].where(out["gsector"].notna(), C.UNCLASSIFIED)
    return out, overlapping


def coverage_by_year(classified: pd.DataFrame) -> pd.DataFrame:
    """By year: universe-months, the share with a GVKEY, the share with a sector, and the Unclassified count."""
    has_sector = classified["sector"] != C.UNCLASSIFIED
    flags = classified["linkprim"] if "linkprim" in classified else pd.Series(None, index=classified.index, dtype=object)
    f = classified.assign(year=pd.PeriodIndex(classified["month"]).year,
                          has_gvkey=classified["gvkey"].notna(), has_sector=has_sector,
                          secondary=has_sector & flags.isin(("J", "N")))
    g = f.groupby("year")
    out = pd.DataFrame({
        "universe_months": g.size(),
        "months_with_gvkey": g["has_gvkey"].sum().astype("int64"),
        "months_with_sector": g["has_sector"].sum().astype("int64"),
        "months_with_sector_through_a_j_or_n_link": g["secondary"].sum().astype("int64"),
    })
    out["months_unclassified"] = out["universe_months"] - out["months_with_sector"]
    out["share_with_gvkey"] = (out["months_with_gvkey"] / out["universe_months"]).round(6)
    out["share_with_sector"] = (out["months_with_sector"] / out["universe_months"]).round(6)
    return out.rename_axis("year").reset_index()


def later_classification_by_year(classified: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """By year, the Unclassified universe-months split by cause: no GVKEY, or a GVKEY with no GICS record in
    force. Of the second, those whose company has a GICS record starting after the last trading day of the
    month before are counted: applying a company's later record to earlier months would fill them, which
    the study does not do, because that record was not known when the month began."""
    last_start = history.groupby("gvkey")["indfrom"].max().rename("last_start")
    f = classified.merge(last_start, left_on="gvkey", right_index=True, how="left")
    f["year"] = pd.PeriodIndex(f["month"]).year
    f["unclassified"] = f["sector"] == C.UNCLASSIFIED
    f["no_gvkey"] = f["unclassified"] & f["gvkey"].isna()
    f["later_record"] = f["unclassified"] & (f["last_start"] > f["date_prev"])
    g = f.groupby("year")
    return pd.DataFrame({
        "months_unclassified": g["unclassified"].sum().astype("int64"),
        "unclassified_without_gvkey": g["no_gvkey"].sum().astype("int64"),
        "unclassified_with_later_record": g["later_record"].sum().astype("int64"),
    }).rename_axis("year").reset_index()


def coverage_by_month(classified: pd.DataFrame) -> pd.DataFrame:
    """By month: the share of the universe's members with a GICS sector, and the share of its capitalisation
    held by members with one, members without a capitalisation left out of the second."""
    f = classified.assign(has_sector=classified["sector"] != C.UNCLASSIFIED)
    f["cap_with_sector"] = f["cap"].where(f["has_sector"])
    g = f.groupby("month")
    out = pd.DataFrame({"members": g.size(), "members_with_sector": g["has_sector"].sum(),
                        "cap": g["cap"].sum(min_count=1), "cap_with_sector": g["cap_with_sector"].sum()})
    out["share_members_with_sector"] = out["members_with_sector"] / out["members"]
    out["share_cap_with_sector"] = out["cap_with_sector"] / out["cap"]
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out[["share_members_with_sector", "share_cap_with_sector"]]


def meets_coverage(by_month: pd.DataFrame) -> pd.Series:
    """True in a month where both shares reach the levels fixed for the evaluation window."""
    return ((by_month["share_members_with_sector"] >= C.EVAL_GICS_MEMBER_SHARE_MIN)
            & (by_month["share_cap_with_sector"] >= C.EVAL_GICS_CAP_SHARE_MIN))


def evaluation_start(by_month: pd.DataFrame, start: str = C.EVAL_START, end: str = C.EVAL_END) -> dict:
    """The evaluation window's start under the coverage condition: the window starts in `start` if that month
    meets both levels, or else in the first later month of the same year that does. The condition holds only if
    every month from that start to `end` meets both levels. When it does not, the study stops for a ruling."""
    first, last = pd.Period(start, "M"), pd.Period(end, "M")
    ok = meets_coverage(by_month)
    candidates = [m for m in ok.index if first <= m and m.year == first.year and ok[m]]
    out = {"start_fixed": start, "end": end, "member_share_level": C.EVAL_GICS_MEMBER_SHARE_MIN,
           "cap_share_level": C.EVAL_GICS_CAP_SHARE_MIN}
    if not candidates:
        return {**out, "start": None, "months": 0, "months_below_after_start": None,
                "first_month_below_after_start": None, "condition_met": False}
    s = candidates[0]
    span = ok[(ok.index >= s) & (ok.index <= last)]
    below = span[~span]
    return {**out, "start": str(s), "months": int(len(span)), "months_below_after_start": int(len(below)),
            "first_month_below_after_start": str(below.index[0]) if len(below) else None,
            "condition_met": bool(len(below) == 0)}


def coverage_by_year_after(by_month: pd.DataFrame, start: str = C.EVAL_START, end: str = C.EVAL_END) -> pd.DataFrame:
    """By year, over the months from `start` to `end`: the lowest and the mean monthly share of members and of
    capitalisation with a sector, and the months below either level. The first year holds only its months
    from `start`."""
    b = by_month[(by_month.index >= pd.Period(start, "M")) & (by_month.index <= pd.Period(end, "M"))]
    below = ~meets_coverage(b)
    g = b.groupby(b.index.year)
    out = pd.DataFrame({
        "months": g.size(),
        "lowest_share_members_with_sector": g["share_members_with_sector"].min().round(6),
        "mean_share_members_with_sector": g["share_members_with_sector"].mean().round(6),
        "lowest_share_cap_with_sector": g["share_cap_with_sector"].min().round(6),
        "mean_share_cap_with_sector": g["share_cap_with_sector"].mean().round(6),
        "months_below_a_level": below.groupby(b.index.year).sum().astype("int64"),
    })
    return out.rename_axis("year").reset_index()


def gics_columns() -> list:
    """The eleven committed sector-weight columns, in GICS code order."""
    return ["weight_" + code for code in C.GICS_SECTORS]


def sector_columns() -> list:
    """The twelve sector-weight columns the study uses in memory, in GICS code order with Unclassified last."""
    return gics_columns() + ["weight_" + C.UNCLASSIFIED.lower()]


def sector_weights(panel: pd.DataFrame, classified_only: bool = False) -> pd.DataFrame:
    """The benchmark's weight in each sector by month, members without a capitalisation left out. By default
    the twelve sectors as shares of the universe's capitalisation. With classified_only, the eleven GICS
    sectors as shares of the classified capitalisation, the form committed. Each row sums to one."""
    p = panel[panel["cap"].notna()]
    if classified_only:
        p = p[p["sector"] != C.UNCLASSIFIED]
    by = p.groupby(["month", "sector"])["cap"].sum().unstack(fill_value=0.0)
    by = by.div(by.sum(axis=1), axis=0)
    by.columns = ["weight_" + (c if c in C.GICS_SECTORS else C.UNCLASSIFIED.lower()) for c in by.columns]
    out = by.reindex(columns=gics_columns() if classified_only else sector_columns(), fill_value=0.0)
    out.index.name = "month"
    return out.reset_index()


def cells_by_month(panel: pd.DataFrame) -> pd.DataFrame:
    """For each month and sector, Unclassified included: the members with a capitalisation, and the largest
    single member's share of the sector's capitalisation. It stays in memory."""
    p = panel[panel["cap"].notna()]
    g = p.groupby(["month", "sector"])["cap"]
    out = pd.DataFrame({"members": g.size(), "largest_share": g.max() / g.sum()})
    return out.reset_index()


def cell_rule(cells: pd.DataFrame, sectors, start: str, end: str) -> dict:
    """The rule for any cell committed by month (constants.SMALL_CELL_MIN_MEMBERS, SMALL_CELL_MAX_MEMBER_SHARE),
    checked for the given sectors from `start` to `end`. A sector with no member in a month has weight zero and
    holds no member's value: it is counted as empty and does not fail."""
    months = pd.period_range(start, end, freq="M")
    grid = pd.MultiIndex.from_product([months, list(sectors)], names=["month", "sector"])
    c = cells.assign(month=pd.PeriodIndex(cells["month"], freq="M")).set_index(["month", "sector"]).reindex(grid)
    held = c["members"].fillna(0) > 0
    fails = held & ((c["members"] < C.SMALL_CELL_MIN_MEMBERS) | (c["largest_share"] > C.SMALL_CELL_MAX_MEMBER_SHARE))
    h = c[held]
    return {"cells": int(len(c)), "empty_cells": int((~held).sum()), "cells_failing": int(fails.sum()),
            "fewest_members": int(h["members"].min()) if len(h) else None,
            "largest_single_share": round(float(h["largest_share"].max()), 6) if len(h) else None,
            "rule_met": bool(fails.sum() == 0)}
