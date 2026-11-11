"""
The data layer assembled from the extraction, for the notebooks: the universe, the monthly panel and the
sectors, each rebuilt in memory from the files on every run, so that no per-security table is ever written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C
from . import extraction as X
from . import returns as R
from . import sectors as S
from . import universe as U


def load_universe(root) -> dict:
    """CRSP's list of members and its monthly index file, the membership by month, and its long form."""
    spells, bad_spells = X.read_table(root, "crsp_dsp500list")
    index_file, bad_index = X.read_table(root, "crsp_msp500")
    ends = U.month_ends(index_file)
    matrix = U.membership_matrix(spells, ends)
    return {"spells": spells, "index_file": index_file, "ends": ends, "matrix": matrix,
            "ulong": U.universe_long(matrix, ends), "unparsed": {**bad_spells, **bad_index}}


def load_panel(root, universe: dict) -> dict:
    """The monthly stock file for the universe's stocks, the delistings, and the monthly panel built from them."""
    monthly, bad_monthly = X.read_table(root, "crsp_msf")
    monthly = monthly[monthly["permno"].isin(universe["matrix"].columns)].reset_index(drop=True)
    delistings, bad_delist = X.read_table(root, "crsp_msedelist")
    panel = R.monthly_panel(universe["ulong"], monthly, delistings)
    return {"monthly": monthly, "delistings": delistings, "panel": panel,
            "unparsed": {**bad_monthly, **bad_delist}}


def load_sectors(root, frame) -> dict:
    """GVKEYs through the sector join's links (any primary flag, in the order P, C, J, N) and GICS by month for
    the rows of `frame` (month, permno, date_prev), with the link rows read, the links kept and the GICS history."""
    raw_links, bad_links = X.read_table(root, "crsp_ccm_lnkhist")
    links = S.sector_links(raw_links)
    linked, multiple = S.link_gvkeys(frame, links)
    history, bad_gics = X.read_table(root, "comp_co_hgic")
    classified, overlapping = S.gics_by_month(linked, history)
    return {"link_rows": int(len(raw_links)), "links": links, "history": history, "linked": linked,
            "classified": classified, "multiple_gvkeys": multiple, "overlapping_gics": overlapping,
            "unparsed": {**bad_links, **bad_gics}}


def daily_years() -> list:
    """The years of the daily stock file, one file each."""
    return list(range(int(C.DAILY_FILE_START[:4]), int(C.CRSP_END[:4]) + 1))


# Rows of a removed stock are kept only this many calendar days either side of its removal. Ten trading
# days never span more than 16 calendar days, so the margin loses nothing gate G3 can use.
NEAR_REMOVAL_DAYS = 40


def daily_pass(root, ulong, removed, years=None, progress: bool = False) -> dict:
    """One pass over the daily stock file, a year at a time, so that no more than one year is in memory.

    Kept from each year: its trading days, the member-days with a return (for the coverage by year), and
    the priced rows of each removed stock within NEAR_REMOVAL_DAYS of its removal (for gate G3). `removed`
    holds permno and ending, one row per removal. Nothing else of the file is kept.
    """
    days, kept, present, unparsed = [], [], {}, {}
    near = pd.Timedelta(days=NEAR_REMOVAL_DAYS)
    for year in (daily_years() if years is None else years):
        d, bad = X.read_table(root, "crsp_dsf", years=[year])
        for column, n in bad.items():
            unparsed[column] = unparsed.get(column, 0) + n
        days.append(np.unique(d["date"].dropna().values))
        present[year] = R.member_days_with_return(d, ulong)
        priced = d.loc[d["permno"].isin(removed["permno"]) & d["prc"].notna(), ["permno", "date", "prc"]]
        close = priced.merge(removed[["permno", "ending"]], on="permno")
        close = close[(close["date"] - close["ending"]).abs() <= near]
        kept.append(close[["permno", "date", "prc"]].drop_duplicates(["permno", "date"]))
        if progress:
            print(year, end=" ", flush=True)
    if progress:
        print()
    return {"calendar": np.sort(np.concatenate(days)), "present": present,
            "removed_rows": pd.concat(kept, ignore_index=True), "unparsed": unparsed}
