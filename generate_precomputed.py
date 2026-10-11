"""Rechnet die vorgerechnete Tabelle der Scheibe (Plätze x Stapeltempo): Wahrheit aus langen Läufen, 30 Wiederholungen je Punkt für zwei Lauflängen.

    python generate_precomputed.py            # schreibt precomputed_slice.json (rund 3 bis 5 Minuten auf 14 Kernen)

Jede Zahl ist eine echte Simulation (bao_network.py); der Zufall kommt aus festen Seeds, die Tabelle ist reproduzierbar.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import bao_constants as C
import bao_scenario as S


def cell(args):
    i, j = args
    k, sp = C.PLACES[i], C.SPEEDS[j]
    base = 10_000 + 1_000 * (i * C.N_SPEEDS + j)
    truth = sum(S.simulate_cost(k, sp, C.TRUTH_TRUCKS, base + r) for r in range(C.TRUTH_RUNS)) / C.TRUTH_RUNS
    reps = {t: [S.simulate_cost(k, sp, t, base + 100 * (1 + C.TRUCK_CHOICES.index(t)) + r) for r in range(C.REPLICATES)] for t in C.TRUCK_CHOICES}
    return i, j, truth, reps


def main(workers=14):
    t0 = time.time()
    jobs = [(i, j) for i in range(C.N_PLACES) for j in range(C.N_SPEEDS)]
    truth = [[0.0] * C.N_SPEEDS for _ in range(C.N_PLACES)]
    reps = {str(t): [[None] * C.N_SPEEDS for _ in range(C.N_PLACES)] for t in C.TRUCK_CHOICES}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, j, tr, rp in ex.map(cell, jobs, chunksize=1):
            truth[i][j] = tr
            for t in C.TRUCK_CHOICES:
                reps[str(t)][i][j] = rp[t]
    out = {"places": list(C.PLACES), "speeds": list(C.SPEEDS), "truth": truth, "reps": reps,
           "meta": {"truth_trucks": C.TRUTH_TRUCKS, "truth_runs": C.TRUTH_RUNS, "replicates": C.REPLICATES, "gamma_per_hour": C.GAMMA_PER_HOUR,
                    "seconds": round(time.time() - t0)}}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_slice.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("geschrieben", path, "in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
