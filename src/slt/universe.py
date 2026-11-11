"""
The universe: the point-in-time S&P 500 from CRSP's own list of index members.

A stock is in the universe in month t if one of its membership spells covers the last trading day of
month t-1 (constants.MEMBERSHIP_RULE), so nothing unknown when the month begins decides membership. This
module builds that membership month by month, compares its count with CRSP's own count for the same day,
finds the first usable month, compares the universe with Compustat's list of members, and counts the
members whose share code is not ordinary common stock. Every function works on frames already read from
the extraction and returns aggregates or frames that stay in memory; nothing here reads or writes a file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C


def month_ends(index_file: pd.DataFrame) -> pd.Series:
    """CRSP's date for each month, which is the month's last trading day, indexed by month and sorted."""
    dates = pd.to_datetime(index_file["caldt"]).sort_values().reset_index(drop=True)
    months = pd.PeriodIndex(dates, freq="M")
    if len(months) > 1 and not (np.diff(months.asi8) == 1).all():
        raise ValueError("CRSP's monthly dates are not one per month without gaps")
    return pd.Series(dates.values, index=months, name="month_end")


def spell_checks(spells: pd.DataFrame) -> dict:
    """Counts that say whether CRSP's list is well formed: spells, PERMNOs, PERMNOs with more than one
    spell, spells that overlap an earlier spell of the same PERMNO, spells that end before they start, and
    spells that end on CRSP's last date, which are the current members."""
    s = spells.sort_values(["permno", "start"])
    previous_end = s.groupby("permno")["ending"].shift()
    per = s.groupby("permno").size()
    return {"spells": int(len(s)), "permnos": int(per.size), "permnos_with_more_than_one_spell": int((per > 1).sum()),
            "overlapping_spells": int((s["start"] <= previous_end).sum()),
            "spells_ending_before_start": int((s["ending"] < s["start"]).sum()),
            "spells_ending_on_last_date": int((s["ending"] == pd.Timestamp(C.CRSP_END)).sum()),
            "earliest_start": str(s["start"].min().date()), "latest_end": str(s["ending"].max().date())}


def membership_matrix(spells: pd.DataFrame, ends: pd.Series) -> pd.DataFrame:
    """True where a stock is in the universe in month t: one of its spells covers the last trading day of
    month t-1, both ends of the spell included.

    Rows are the months t from the month after CRSP's first date to the month of its last date; columns
    are PERMNOs. A stock with more than one spell is in the universe in every month one of them covers.
    """
    prev = ends.values[:-1]                       # the last trading day of month t-1, for each month t
    months = ends.index[1:]
    permnos = np.sort(spells["permno"].unique())
    column = {p: i for i, p in enumerate(permnos)}
    member = np.zeros((len(months), len(permnos)), dtype=bool)
    first = np.searchsorted(prev, spells["start"].values, side="left")     # first month-end on or after the start
    after = np.searchsorted(prev, spells["ending"].values, side="right")   # first month-end after the end
    for permno, a, b in zip(spells["permno"].values, first, after):
        if b > a:
            member[a:b, column[permno]] = True
    return pd.DataFrame(member, index=months, columns=permnos)


def universe_long(matrix: pd.DataFrame, ends: pd.Series) -> pd.DataFrame:
    """The membership as one row per stock and month: month, permno, and the last trading day of the month before."""
    rows, cols = np.nonzero(matrix.values)
    out = pd.DataFrame({"month": matrix.index[rows], "permno": matrix.columns[cols].astype("int64")})
    out["date_prev"] = ends.reindex(pd.PeriodIndex(out["month"]) - 1).values
    return out.sort_values(["month", "permno"], ignore_index=True)


def counts_against_crsp(matrix: pd.DataFrame, index_file: pd.DataFrame) -> pd.DataFrame:
    """The universe's count in month t beside CRSP's count on its row for month t-1, the same day."""
    crsp = index_file.assign(month=pd.PeriodIndex(pd.to_datetime(index_file["caldt"]), freq="M")).set_index("month")
    out = pd.DataFrame({"universe_count": matrix.sum(axis=1).astype("int64")}, index=matrix.index)
    out["crsp_count"] = crsp["totcnt"].reindex(out.index - 1).values
    out["difference"] = out["universe_count"] - out["crsp_count"]
    return out


def usable_window(counts: pd.DataFrame, within: int = C.USABLE_COUNT_WITHIN) -> dict:
    """The data layer's usable window: from the first month in which the count is within `within` names of
    CRSP's count to the last month, with the verdict of gate G2."""
    close = counts["difference"].abs() <= within
    if not close.any():
        return {"first_month": None, "last_month": str(counts.index[-1]), "months": 0, "years": 0.0,
                "g2_level_years": C.G2_USABLE_YEARS_MIN, "g2_met": False}
    first = close.idxmax()
    last = counts.index[-1]
    months = int((last - first).n) + 1
    years = months / 12
    return {"first_month": str(first), "last_month": str(last), "months": months, "years": round(years, 2),
            "g2_level_years": C.G2_USABLE_YEARS_MIN, "g2_met": bool(years >= C.G2_USABLE_YEARS_MIN)}


def count_check_by_year(counts: pd.DataFrame, within: int = C.USABLE_COUNT_WITHIN) -> pd.DataFrame:
    """The count comparison summarised by year, in the two figures committed: the mean difference and the
    months outside the tolerance. A count of months with any difference is left out, because in a year
    with none it would give CRSP's count month by month."""
    diff = counts["difference"]
    by = diff.groupby(counts.index.year)
    return pd.DataFrame({
        "months": by.size(),
        "mean_count_difference": by.mean().round(4),
        "months_outside_tolerance": (diff.abs() > within).groupby(counts.index.year).sum().astype("int64"),
    }).rename_axis("year").reset_index()


def index_rows(history: pd.DataFrame, index_key: str = C.SP500_GVKEYX) -> pd.DataFrame:
    """Compustat's rows of its constituent history for one index, selected by the index's key."""
    return history[history["gvkeyx"] == index_key]


def compustat_members_long(history: pd.DataFrame, ends: pd.Series, index_key: str = C.SP500_GVKEYX) -> pd.DataFrame:
    """Compustat's list of index members by month, on the same rule as the universe: month t holds the GVKEYs
    whose interval covers the last trading day of month t-1. A missing end date means a current member."""
    rows = index_rows(history, index_key)
    prev = ends.values[:-1]
    months = ends.index[1:]
    thru = rows["thru"].fillna(pd.Timestamp("2099-12-31"))
    first = np.searchsorted(prev, rows["from"].values, side="left")
    after = np.searchsorted(prev, thru.values, side="right")
    parts = [pd.DataFrame({"month": months[a:b], "gvkey": g}) for g, a, b in zip(rows["gvkey"].values, first, after) if b > a]
    if not parts:
        return pd.DataFrame({"month": pd.PeriodIndex([], freq="M"), "gvkey": pd.Series([], dtype=object)})
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def list_agreement_by_year(universe_gvkeys: pd.DataFrame, compustat: pd.DataFrame) -> pd.DataFrame:
    """CRSP's universe against Compustat's list, month by month, summed by year.

    universe_gvkeys holds month and gvkey for the universe members that have a GVKEY. Two readings are
    reported: the share of Compustat's members that are in the universe, and the share of the two lists'
    union on which they disagree. Members without a GVKEY are counted separately by the caller.
    """
    a = universe_gvkeys.dropna(subset=["gvkey"])[["month", "gvkey"]].drop_duplicates().assign(in_crsp=True)
    b = compustat[["month", "gvkey"]].drop_duplicates().assign(in_compustat=True)
    both = a.merge(b, on=["month", "gvkey"], how="outer")
    both["in_crsp"] = both["in_crsp"].eq(True)
    both["in_compustat"] = both["in_compustat"].eq(True)
    both["year"] = pd.PeriodIndex(both["month"]).year
    both["in_both"] = both["in_compustat"] & both["in_crsp"]
    both["crsp_only"] = both["in_crsp"] & ~both["in_compustat"]
    both["compustat_only"] = both["in_compustat"] & ~both["in_crsp"]
    g = both.groupby("year")
    out = pd.DataFrame({
        "union_months": g.size(),
        "compustat_months": g["in_compustat"].sum().astype("int64"),
        "compustat_months_in_universe": g["in_both"].sum().astype("int64"),
        "universe_months_not_in_compustat": g["crsp_only"].sum().astype("int64"),
        "compustat_months_not_in_universe": g["compustat_only"].sum().astype("int64"),
    })
    out["share_of_compustat_in_universe"] = (out["compustat_months_in_universe"] / out["compustat_months"]).round(6)
    # The definition of 9 November. On this subscription Compustat's table holds current members only, so this
    # share counts every name that left the index before September 2026, and the column is named for that.
    out["two_sided_share_counting_exits_compustat_omits"] = (
        (out["universe_months_not_in_compustat"] + out["compustat_months_not_in_universe"]) / out["union_months"]).round(6)
    return out.rename_axis("year").reset_index()


def compustat_rows_summary(rows: pd.DataFrame, links: pd.DataFrame) -> dict:
    """Counts beside the cross-check: Compustat's rows for the index, its companies, the rows that joined after
    CRSP's last date, and, of the others, the rows whose GVKEY has no primary link valid at any time from the
    join to CRSP's last date. `links` are the primary links (sectors.primary_links)."""
    end = pd.Timestamp(C.CRSP_END)
    inside = rows[rows["from"] <= end].reset_index(drop=True)
    m = inside.reset_index().merge(links[["gvkey", "linkdt", "linkenddt"]], on="gvkey", how="left")
    valid = (m["linkdt"] <= end) & (m["linkenddt"] >= m["from"])
    linked_rows = set(m.loc[valid, "index"])
    return {"rows": int(len(rows)), "companies": int(rows["gvkey"].nunique()),
            "rows_joined_after_last_date": int((rows["from"] > end).sum()),
            "rows_without_a_primary_link": int(len(inside) - len(linked_rows))}


def join_dates(rows: pd.DataFrame, links: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """For each of Compustat's rows that joined by CRSP's last date: the months from Compustat's join date to the
    start of CRSP's membership interval that covers Compustat's months, through the primary links of its GVKEY.

    The interval taken is the CRSP spell, of any PERMNO linked to the GVKEY over Compustat's months, that overlaps
    those months the longest. A positive figure means CRSP dates the join later. A row with no such interval has
    no figure. The frame stays in memory: only the counts by year are committed.
    """
    end = pd.Timestamp(C.CRSP_END)
    r = rows[rows["from"] <= end][["gvkey", "from"]].reset_index(drop=True).reset_index().rename(columns={"index": "row"})
    m = r.merge(links[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="inner")
    m = m[(m["linkdt"] <= end) & (m["linkenddt"] >= m["from"])]
    m = m.merge(spells[["permno", "start", "ending"]], on="permno", how="inner")
    lo = np.maximum(m["start"].values, m["from"].values)
    hi = np.minimum(m["ending"].values, np.datetime64(end))
    m["overlap_days"] = (hi - lo).astype("timedelta64[D]").astype("int64")
    m = m[m["overlap_days"] >= 0]
    m = m.sort_values(["row", "overlap_days", "start"], ascending=[True, False, False]).drop_duplicates("row")
    out = r.merge(m[["row", "start"]], on="row", how="left")
    a, b = pd.PeriodIndex(out["from"], freq="M"), out["start"]
    has = b.notna()
    out["months_apart"] = np.nan
    out.loc[has, "months_apart"] = (pd.PeriodIndex(b[has], freq="M").asi8 - a[has.values].asi8).astype(float)
    return out


def join_dates_by_year(joined: pd.DataFrame, within: int = C.CROSS_CHECK_JOIN_WITHIN_MONTHS) -> pd.DataFrame:
    """By the year Compustat dates the join: the members whose join both lists date to the same month, within
    `within` months, further apart with CRSP's date earlier or later, and those with no CRSP interval."""
    d = joined["months_apart"]
    f = joined.assign(year=pd.PeriodIndex(joined["from"], freq="M").year,
                      same=d == 0, near=d.abs().between(1, within), earlier=d < -within, later=d > within, none=d.isna())
    g = f.groupby("year")
    return pd.DataFrame({
        "members_joined": g.size(),
        "same_month": g["same"].sum().astype("int64"),
        "within_one_month": g["near"].sum().astype("int64"),
        "further_apart_crsp_earlier": g["earlier"].sum().astype("int64"),
        "further_apart_crsp_later": g["later"].sum().astype("int64"),
        "no_crsp_interval": g["none"].sum().astype("int64"),
    }).rename_axis("year").reset_index()


def share_codes(ulong: pd.DataFrame, names: pd.DataFrame) -> tuple:
    """The share code of each universe member in each month, from the names record in force on the last
    trading day of the month before. Returns the frame and the count of stock-months matched by more than one record."""
    m = ulong.merge(names[["permno", "namedt", "nameendt", "shrcd"]], on="permno", how="inner")
    m = m[(m["namedt"] <= m["date_prev"]) & (m["date_prev"] <= m["nameendt"])]
    per = m.groupby(["permno", "month"]).size()
    overlaps = int((per > 1).sum())
    m = m.sort_values(["permno", "month", "namedt"], ascending=[True, True, False]).drop_duplicates(["permno", "month"])
    out = ulong.merge(m[["permno", "month", "shrcd"]], on=["permno", "month"], how="left")
    return out, overlaps


def share_codes_by_year(coded: pd.DataFrame) -> pd.DataFrame:
    """Universe members outside the common share codes, and members with no share code, by year."""
    common = coded["shrcd"].isin(C.SHARE_CODES_COMMON)
    missing = coded["shrcd"].isna()
    frame = coded.assign(year=pd.PeriodIndex(coded["month"]).year, outside=~common & ~missing, missing=missing)
    g = frame.groupby("year")
    return pd.DataFrame({
        "universe_months": g.size(),
        "months_outside_common_codes": g["outside"].sum().astype("int64"),
        "months_without_share_code": g["missing"].sum().astype("int64"),
        "names_outside_common_codes": frame[frame["outside"]].groupby("year")["permno"].nunique()
                                      .reindex(g.size().index, fill_value=0).astype("int64"),
    }).rename_axis("year").reset_index()


def share_code_totals(coded: pd.DataFrame) -> pd.DataFrame:
    """Universe-months by share code over the whole sample."""
    codes = coded["shrcd"].astype("Int64").astype(object).where(coded["shrcd"].notna(), "none")
    return codes.value_counts().rename_axis("share_code").reset_index(name="universe_months")
