"""Vorgerechnete Studien (rund 15 bis 25 Minuten auf 14 Kernen):

1. Scheibe (Plätze x Stapeltempo, Tabelle): je Lauflänge 20 feste Läufe, vier Verfahren, Budget bis C.N_MAX, wahre Kosten der Empfehlung je Schritt;
   dazu die Kalibrierung des Gauß-Prozesses (Abdeckung des 95-%-Intervalls) über das Budget.
2. Volles Netz (sechs Entscheidungen, echte Simulationen à 15 000 Lkw, Kandidatenmenge gerasterter Sobol-Punkte): vier Verfahren, 20 Läufe, Budgets
   10/20/30/40; die Empfehlung wird in einer langen Simulation (2 x 100 000 Lkw) nachbewertet. Dazu die Formel ohne Platzgrenze.

    python generate_study.py            # schreibt precomputed_study.json
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import bao_algorithm as A
import bao_benchmarks as B
import bao_constants as C
import bao_evaluation as ev
import bao_scenario as S

METHOD_SEED_OFFSET = {m: 1000 * (k + 1) for k, m in enumerate(C.METHODS)}


def run_method(method, cand, ev, n, rng, n_start, bandwidth=0.15):
    """Ein Verfahren auf einer Kandidatenmenge; gleiche Aufrufstelle für Scheibe und volles Netz."""
    if method == "random":
        return A.random_search(len(cand), ev, n, rng)
    if method == "tpe":
        return A.tpe_search(cand, ev, n, rng, n_start=n_start, bandwidth=bandwidth)
    return A.bo_search(cand, ev, n, rng, n_start=n_start, log=(method == "bo_log"))


def landscape_run(args):
    """Ein Lauf auf Branin (kind='branin', noise) oder der Terminal-Scheibe (kind='terminal', trucks): wahre Werte der Empfehlung je Budget."""
    kind, method, param, seed = args
    st = ev.Settings(n=C.N_MAX, noise=param if kind == "branin" else C.DEFAULT_BRANIN_NOISE, trucks=param if kind == "terminal" else C.DEFAULT_TRUCKS, seed=seed)
    return kind, method, param, seed, ev.run_one(kind, method, st, C.N_MAX).truth_trace


def calib_run(args):
    n, seed = args
    pts = S.grid_points()
    cand = S.to_unit(pts)
    term = S.Terminal(15000, seed + 55)
    rng = np.random.default_rng(seed + 99)
    idx = [int(i) for i in rng.choice(len(cand), size=n, replace=False)]
    y = [term.observe(int(pts[i][0]), int(pts[i][1])) for i in idx]
    gp = A.GaussianProcess().fit(cand[idx], y)
    mu, sd = gp.predict(cand)
    truth = np.array([S.Terminal.truth(int(p[0]), int(p[1])) for p in pts])
    return n, seed, float((np.abs(truth - mu) <= 1.96 * sd).mean()), float(np.mean(np.abs(truth - mu)))


def full_run(args):
    method, seed = args
    pool = S.full_pool()
    counter = {"n": 0}
    base = seed * 10_000 + METHOD_SEED_OFFSET[method]

    def ev(i):
        counter["n"] += 1
        return S.full_cost(*S.full_decode(pool[i]), C.STUDY_TRUCKS, base + counter["n"])

    idx, y = run_method(method, pool, ev, max(C.STUDY_BUDGETS), np.random.default_rng(base), 10, bandwidth=0.25)
    rec = {}
    for b in C.STUDY_BUDGETS:
        d = S.full_decode(pool[idx[int(np.argmin(y[:b]))]])
        rec[b] = {"point": list(d), "long_cost": float(np.mean([S.full_cost(*d, C.TRUTH_TRUCKS, 90_000 + r) for r in range(C.TRUTH_RUNS)]))}
    return method, seed, rec


def formula_pick(rule):
    """Formel-Optimum auf der Kandidatenmenge; mit rule=True nur Punkte mit Plätzen = Stapelgeräte (Faustregel eines Kenners)."""
    pool = S.full_pool()
    pts = [S.full_decode(p) for p in pool]
    vals = np.array([S.full_formula_cost(*d) if (not rule or d[3] == max(d[2], C.PLACES[0])) else np.inf for d in pts])
    d = pts[int(np.argmin(vals))]
    long_cost = float(np.mean([S.full_cost(*d, C.TRUTH_TRUCKS, 90_000 + r) for r in range(C.TRUTH_RUNS)]))
    return list(d), float(vals.min()), long_cost


def bench_run(args):
    method, seed = args
    pool = bench_pool()
    base = 77_000 + seed
    n = max(C.BENCH_BUDGETS)
    ev = lambda i: B.hartmann6(pool[i])
    rng = np.random.default_rng(base + 13 * C.BENCH_METHODS.index(method))
    if method == "random":
        idx, y = A.random_search(len(pool), ev, n, rng)
    elif method == "tpe":
        idx, y = A.tpe_search(pool, ev, n, rng, n_start=10, bandwidth=0.25)
    else:
        idx, y = A.bo_search(pool, ev, n, rng, n_start=10, noise=1e-6)
    return method, seed, {b: float(min(y[:b])) for b in C.BENCH_BUDGETS}


def bench_pool():
    from scipy.stats import qmc
    return qmc.Sobol(6, scramble=True, seed=5).random(C.BENCH_POOL)


def main(workers=14):
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs = [("terminal", m, t, s) for t in C.TRUCK_CHOICES for m in C.METHODS for s in C.SWEEP_SEEDS]
        jobs += [("branin", m, z, s) for z in C.BRANIN_NOISES for m in C.METHODS_BY_KIND["branin"] for s in C.SWEEP_SEEDS]
        res = list(ex.map(landscape_run, jobs, chunksize=1))
        out["slice"] = {str(t): {m: [r[4] for r in res if r[0] == "terminal" and r[1] == m and r[2] == t] for m in C.METHODS} for t in C.TRUCK_CHOICES}
        out["branin"] = {str(float(z)): {m: [r[4] for r in res if r[0] == "branin" and r[1] == m and r[2] == z] for m in C.METHODS_BY_KIND["branin"]} for z in C.BRANIN_NOISES}
        print("Scheibe fertig", round(time.time() - t0), "s", flush=True)
        cres = list(ex.map(calib_run, [(n, s) for n in C.N_SWEEP for s in C.SWEEP_SEEDS], chunksize=1))
        out["calibration"] = {str(n): {"coverage": [r[2] for r in cres if r[0] == n], "mae": [r[3] for r in cres if r[0] == n]} for n in C.N_SWEEP}
        print("Kalibrierung fertig", round(time.time() - t0), "s", flush=True)
        fres = list(ex.map(full_run, [(m, s) for m in C.METHODS for s in C.STUDY_SEEDS], chunksize=1))
        out["full"] = {m: {str(b): [r[2][b]["long_cost"] for r in fres if r[0] == m] for b in C.STUDY_BUDGETS} for m in C.METHODS}
        out["full_points"] = {m: [r[2][max(C.STUDY_BUDGETS)]["point"] for r in fres if r[0] == m] for m in C.METHODS}
    for key, rule in (("formula", False), ("formula_rule", True)):
        fp = formula_pick(rule)
        out[key] = {"point": fp[0], "formula_value": fp[1], "long_cost": fp[2]}
    with ProcessPoolExecutor(max_workers=workers) as ex2:
        bres = list(ex2.map(bench_run, [(m, s) for m in C.BENCH_METHODS for s in C.STUDY_SEEDS], chunksize=1))
    pool = bench_pool()
    opt = min(B.hartmann6(p) for p in pool)
    out["bench"] = {"optimum": float(opt), "regret": {m: {str(b): [r[2][b] - opt for r in bres if r[0] == m] for b in C.BENCH_BUDGETS} for m in C.BENCH_METHODS}}
    out["meta"] = {"pool": int(len(S.full_pool())), "seeds": len(C.SWEEP_SEEDS), "study_seeds": len(C.STUDY_SEEDS), "seconds": round(time.time() - t0)}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_study.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("fertig in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
