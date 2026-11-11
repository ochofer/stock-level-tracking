"""
Returns and market capitalisation of the universe members, with delisting returns applied.

Monthly returns come from CRSP's monthly stock file. A delisting in the month is applied as data rule D4
says, after Shumway (1997): CRSP's delisting return where CRSP gives one; where it is missing, minus 30 per
cent for a performance-related code (400 to 599) and zero for any other code, each such month counted.
CRSP's code 100 marks a security still active and is not a delisting. A member with no return and no
delisting in the month is counted and left out of that month's benchmark return. The capitalisation at the
start of month t is the absolute price times shares outstanding at the end of month t-1, in thousands of
dollars; a missing or non-positive value is missing. Nothing here reads or writes a file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C

ACTIVE_CODE = 100   # CRSP's code for a security that has not been delisted


def monthly_panel(ulong: pd.DataFrame, monthly: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    """One row per universe member and month: its return and price-only return with any delisting applied,
    its start-of-month capitalisation, and how the month's delisting, if any, was treated."""
    s = monthly.assign(month=pd.PeriodIndex(monthly["date"], freq="M"))
    s = s.drop_duplicates(["permno", "month"], keep="last")
    current = s[["permno", "month", "ret", "retx"]]
    previous = s[["permno", "month", "prc", "shrout"]].copy()
    previous["month"] = previous["month"] + 1
    previous["cap"] = previous["prc"].abs() * previous["shrout"]
    previous.loc[~(previous["cap"] > 0), "cap"] = np.nan
    p = (ulong.merge(current, on=["permno", "month"], how="left")
              .merge(previous[["permno", "month", "cap"]], on=["permno", "month"], how="left"))
    d = delistings[delistings["dlstcd"].notna() & (delistings["dlstcd"] != ACTIVE_CODE)].copy()
    d["month"] = pd.PeriodIndex(d["dlstdt"], freq="M")
    d = d.drop_duplicates(["permno", "month"], keep="last")
    p = p.merge(d[["permno", "month", "dlstcd", "dlret", "dlretx"]], on=["permno", "month"], how="left")
    delisted = p["dlstcd"].notna().values
    performance = p["dlstcd"].between(*C.DELIST_PERFORMANCE_CODES).values
    for given, used in (("dlret", "dlret_used"), ("dlretx", "dlretx_used")):
        missing_value = np.where(performance, C.DELIST_PERFORMANCE_MISSING_RETURN, C.DELIST_OTHER_MISSING_RETURN)
        p[used] = np.where(~delisted, np.nan, np.where(p[given].notna(), p[given], missing_value))
    p["delisting"] = np.select([~delisted, p["dlret"].notna().values, performance], ["none", "given", "shumway"], default="zero")
    for r, used, out in (("ret", "dlret_used", "ret_total"), ("retx", "dlretx_used", "retx_total")):
        compounded = (1 + p[r]) * (1 + p[used]) - 1
        p[out] = np.where(delisted, np.where(p[r].notna(), compounded, p[used]), p[r])
    p["return_missing"] = p["ret_total"].isna() & ~delisted
    return p


def delisting_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    """Data rule D4's counts by year: delisting months by treatment, and members with no return and no delisting."""
    f = panel.assign(year=pd.PeriodIndex(panel["month"]).year)
    g = f.groupby("year")
    return pd.DataFrame({
        "universe_months": g.size(),
        "delisting_months": g["delisting"].apply(lambda s: int((s != "none").sum())),
        "delisting_return_given": g["delisting"].apply(lambda s: int((s == "given").sum())),
        "delisting_return_minus_30": g["delisting"].apply(lambda s: int((s == "shumway").sum())),
        "delisting_return_zero": g["delisting"].apply(lambda s: int((s == "zero").sum())),
        "months_without_return": g["return_missing"].sum().astype("int64"),
        "months_without_cap": g["cap"].apply(lambda s: int(s.isna().sum())),
    }).rename_axis("year").reset_index()


def delisting_code_families(panel: pd.DataFrame) -> pd.Series:
    """Delisting months of members by the hundreds of their code: 200 for codes 200 to 299, and so on."""
    codes = panel["dlstcd"].dropna().astype(int)
    out = (codes // 100 * 100).value_counts().sort_index()
    out.index.name = "code_family"
    return out.rename("delisting_months")


def cap_coverage(panel: pd.DataFrame, first_month: str) -> dict:
    """Gate G4: the share of names whose capitalisation is present in at least G4_MONTH_SHARE_MIN of their
    membership months over the usable window, against G4_NAME_SHARE_MIN."""
    p = panel[pd.PeriodIndex(panel["month"]) >= pd.Period(first_month, "M")]
    per_name = p.groupby("permno")["cap"].apply(lambda s: s.notna().mean())
    share = float((per_name >= C.G4_MONTH_SHARE_MIN).mean()) if len(per_name) else 0.0
    return {"names": int(len(per_name)), "names_meeting_month_share": int((per_name >= C.G4_MONTH_SHARE_MIN).sum()),
            "share_of_names": round(share, 6), "month_share_level": C.G4_MONTH_SHARE_MIN,
            "name_share_level": C.G4_NAME_SHARE_MIN, "g4_met": bool(share >= C.G4_NAME_SHARE_MIN)}


def removals(spells: pd.DataFrame) -> pd.DataFrame:
    """Every exit from the universe: a spell that ends before CRSP's last date. A spell ending on that date
    is a current member and not a removal."""
    r = spells[spells["ending"] < pd.Timestamp(C.CRSP_END)][["permno", "ending"]]
    return r.drop_duplicates().reset_index(drop=True)


def removal_windows(removed: pd.DataFrame, calendar: np.ndarray, days: int = C.G3_TRADING_DAYS) -> pd.DataFrame:
    """For each removal in the daily file's span, the first and last trading day within `days` trading days of it."""
    cal = np.sort(np.asarray(calendar, dtype="datetime64[ns]"))
    r = removed[(removed["ending"] >= pd.Timestamp(C.G3_WINDOW[0])) & (removed["ending"] <= pd.Timestamp(C.G3_WINDOW[1]))].copy()
    at = np.searchsorted(cal, r["ending"].values.astype("datetime64[ns]"), side="right") - 1   # the last trading day on or before the removal
    at = np.clip(at, 0, len(cal) - 1)
    r["window_start"] = cal[np.clip(at - days, 0, len(cal) - 1)]
    r["window_end"] = cal[np.clip(at + days, 0, len(cal) - 1)]
    return r.reset_index(drop=True)


def priced_near(windows: pd.DataFrame, daily: pd.DataFrame) -> set:
    """The removals (PERMNO and removal date) that have a price in the daily rows given, inside their window."""
    m = windows.merge(daily[daily["prc"].notna()][["permno", "date"]], on="permno", how="inner")
    m = m[(m["date"] >= m["window_start"]) & (m["date"] <= m["window_end"])]
    return set(zip(m["permno"].tolist(), m["ending"].tolist()))


def g3_verdict(windows: pd.DataFrame, found: set) -> dict:
    """Gate G3 on the removals in the daily file's span."""
    n = len(windows)
    hit = sum(1 for k in zip(windows["permno"].tolist(), windows["ending"].tolist()) if k in found)
    share = hit / n if n else 0.0
    return {"removals": n, "removals_with_price": hit, "share": round(share, 6), "trading_days": C.G3_TRADING_DAYS,
            "share_level": C.G3_SHARE_MIN, "window": list(C.G3_WINDOW), "g3_met": bool(n and share >= C.G3_SHARE_MIN)}


def removals_before_daily_file(removed: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    """Removals before the daily file, reported with no verdict: the share with a price at the end of the
    removal month or of the month before, on the monthly file."""
    r = removed[removed["ending"] < pd.Timestamp(C.G3_WINDOW[0])].copy()
    r["m0"] = pd.PeriodIndex(r["ending"], freq="M")
    priced = monthly[monthly["prc"].notna()].assign(month=lambda d: pd.PeriodIndex(d["date"], freq="M"))[["permno", "month"]]
    keys = set(zip(priced["permno"].tolist(), priced["month"].tolist()))
    hit = sum(1 for p, m in zip(r["permno"].tolist(), r["m0"].tolist()) if (p, m) in keys or (p, m - 1) in keys)
    n = len(r)
    return {"removals": n, "removals_with_price": hit, "share": round(hit / n, 6) if n else None,
            "reading": C.G3_BEFORE_WINDOW}


def member_days_with_return(daily: pd.DataFrame, ulong: pd.DataFrame) -> int:
    """For rows of the daily file: the member-days with a return. A member-day is a trading day of a month
    in which the stock is in the universe."""
    d = daily.assign(month=pd.PeriodIndex(daily["date"], freq="M"))
    m = d.merge(ulong[["permno", "month"]], on=["permno", "month"], how="inner")
    return int(m["ret"].notna().sum())


def trading_days_by_month(calendar) -> pd.Series:
    """The number of trading days in each month of the daily file's calendar."""
    cal = pd.DatetimeIndex(np.unique(np.asarray(calendar, dtype="datetime64[ns]")))
    return pd.Series(1, index=cal).groupby(pd.PeriodIndex(cal, freq="M")).sum()


def daily_coverage_by_year(present: dict, days_by_month: pd.Series, ulong: pd.DataFrame) -> pd.DataFrame:
    """Member-days expected (members in the month times its trading days) against member-days with a return, by year."""
    members = ulong.groupby("month").size()
    common = members.index.intersection(days_by_month.index)
    expected = (members.reindex(common) * days_by_month.reindex(common)).groupby(pd.PeriodIndex(common).year).sum()
    out = pd.DataFrame({"member_days_expected": expected.astype("int64")})
    out["member_days_with_return"] = pd.Series(present).reindex(out.index).fillna(0).astype("int64")
    out["share_with_return"] = (out["member_days_with_return"] / out["member_days_expected"]).round(6)
    return out.rename_axis("year").reset_index()
