"""
The rolling evaluation of the 24 paths, run on the machine that holds the extraction.

    python3 scripts/run_evaluation.py              # the four estimators' groups in four processes, side by side
    python3 scripts/run_evaluation.py --estimator ledoit_wolf   # one group (what each process runs)

Each group builds its inputs from the extraction in place, then runs every path of its estimator month by month
(constants.PATHS), writing each path's state after every month to outputs/evaluation/, which git ignores. A run
that stops is started again with the same command and resumes from the last month written. At the end each path
writes its records by month and its weights by security to the same folder, for notebooks 09 to 11. Nothing in
outputs/ is committed or leaves the machine; the notebooks commit the aggregates.

The linear algebra runs on LINEAR_ALGEBRA_THREADS threads in each process, recorded in the provenance file beside
the versions of cvxpy, Clarabel and NumPy, so that a rerun can be compared like for like.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

LINEAR_ALGEBRA_THREADS = 2
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "evaluation"


def one_group(estimator: str, months: int | None) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    import pandas as pd
    from slt import constants as C
    from slt import evaluation as V
    from slt import extraction as X
    root = X.data_root()
    X.check_manifest(root)
    t0 = time.time()
    print(f"{estimator}: reading the extraction", flush=True)
    inp = V.inputs(root)
    print(f"{estimator}: inputs ready in {time.time() - t0:.0f} s", flush=True)
    window = pd.period_range(C.EVAL_START, C.EVAL_END, freq="M")
    V.run_group(estimator, inp, OUT, months=window[:months] if months else window)
    print(f"{estimator}: done in {time.time() - t0:.0f} s", flush=True)


def all_groups(months: int | None) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from slt import constants as C
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **{v: str(LINEAR_ALGEBRA_THREADS) for v in THREAD_VARIABLES})
    started = datetime.now(timezone.utc)
    procs = []
    for est in C.COV_ESTIMATORS:
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--estimator", est]
        if months:
            cmd += ["--months", str(months)]
        log = open(OUT / f"log_{est}.txt", "a")
        procs.append((est, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"started {est}; its progress is in {OUT.relative_to(ROOT)}/log_{est}.txt", flush=True)
    codes = {}
    for est, p, log in procs:
        codes[est] = p.wait()
        log.close()
        print(f"{est}: finished with code {codes[est]}", flush=True)
    sys.path.insert(0, str(ROOT / "src"))
    from slt import evaluation as V
    record = {"started_utc": started.isoformat(timespec="seconds"),
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "exit_codes": codes, "months_run": months or "the evaluation window", **V.provenance(LINEAR_ALGEBRA_THREADS)}
    (OUT / "provenance.json").write_text(json.dumps(record, indent=1))
    if any(codes.values()):
        raise SystemExit("A group stopped with an error: its last lines are in its log. Run the same command again to resume.")
    print("all four groups finished; the records are in outputs/evaluation/", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--estimator", default=None)
    ap.add_argument("--months", type=int, default=None, help="run only the first N months (a test)")
    a = ap.parse_args()
    if a.estimator:
        one_group(a.estimator, a.months)
    else:
        all_groups(a.months)
