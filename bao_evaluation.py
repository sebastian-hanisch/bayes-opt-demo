"""Auswertung: ein Lauf aller Verfahren auf einer Landschaft (Branin oder Terminal-Scheibe), Kennzahlen, vorgerechnete Studien."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import bao_algorithm as A
import bao_constants as C
import bao_scenario as S

_METHOD_OFFSET = {m: 1000 * (k + 1) for k, m in enumerate(C.METHODS)}
KINDS = ("branin", "terminal")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    noise: float = C.DEFAULT_BRANIN_NOISE        # Branin: Standardabweichung des Beobachtungsrauschens
    trucks: int = C.DEFAULT_TRUCKS               # Terminal: Lkw je Auswertung
    seed: int = C.DEFAULT_SEED


@dataclass
class Run:
    method: str
    idx: list            # Gitterindex je Auswertung (Zeilen-Spalten-Reihenfolge von grid_points)
    y: list              # beobachtete Werte je Auswertung
    truth_trace: list    # wahrer Wert der Empfehlung (beste Beobachtung) nach t Auswertungen


@dataclass
class Analysis:
    kind: str
    settings: Settings
    runs: dict
    truth: np.ndarray    # wahrer Wert je Gitterpunkt [Zeile, Spalte]
    optimum: tuple       # (Zeile, Spalte, Wert)
    x_values: list       # Werte der Spalten
    y_values: list       # Werte der Zeilen
    formula: tuple | None = None   # Terminal: Formel ohne Platzgrenze
    rule: tuple | None = None      # Terminal: Formel + Faustregel


# --- Landschaften ---------------------------------------------------------------------------------------------------------------------------

def make_field(kind, settings, seed):
    """Zielfunktion der Landschaft: (Gitterform, Wahrheit, observe(i, j))."""
    if kind == "branin":
        f = S.BraninField(settings.noise, seed)
        return f
    return S.Terminal(settings.trucks, seed)


def truth_grid(kind):
    return S.BraninField.truth_grid() if kind == "branin" else S.table()["truth"]


def axes(kind):
    if kind == "branin":
        x1, x2, _ = S.BraninField.axes()
        return list(map(float, x1)), list(map(float, x2))
    return list(C.SPEEDS), list(C.PLACES)


def optimum(kind="terminal"):
    t = truth_grid(kind)
    i, j = np.unravel_index(int(np.argmin(t)), t.shape)
    return int(i), int(j), float(t[i, j])


def formula_choice():
    """Das Optimum der Formel (kennt die Platzgrenze nicht: jeder Platz kostet nur) und seine wahren Kosten."""
    vals = np.array([[S.formula_cost(k, sp) for sp in C.SPEEDS] for k in C.PLACES])
    i, j = np.unravel_index(int(np.argmin(vals)), vals.shape)
    return int(i), int(j), float(truth_grid("terminal")[i, j])


def rule_choice():
    """Formel plus Faustregel eines Kenners: so viele Plätze wie Stapelgeräte; das Tempo wählt die Formel. Ohne einen einzigen Lauf."""
    i = C.PLACES.index(C.STAPEL_SERVERS)
    j = int(np.argmin([S.formula_cost(C.PLACES[i], sp) for sp in C.SPEEDS]))
    return int(i), int(j), float(truth_grid("terminal")[i, j])


def run_one(kind, method, settings, n):
    """Ein Suchlauf mit Auswertungen aus der Landschaft; Zufallsströme je Verfahren getrennt und reproduzierbar."""
    truth = truth_grid(kind)
    pts = S.grid_points(truth.shape)
    cand = S.to_unit(pts, truth.shape)
    seed = settings.seed + _METHOD_OFFSET[method]
    field = make_field(kind, settings, seed)
    rng = np.random.default_rng(seed + 7)

    def ev(i):
        return field.observe(int(pts[i][0]), int(pts[i][1]))

    if method == "random":
        idx, y = A.random_search(len(cand), ev, n, rng)
    elif method == "tpe":
        idx, y = A.tpe_search(cand, ev, n, rng, n_start=C.N_START)
    else:
        idx, y = A.bo_search(cand, ev, n, rng, n_start=C.N_START, log=(method == "bo_log"))
    trace = []
    for t in range(1, n + 1):
        p = pts[idx[int(np.argmin(y[:t]))]]
        trace.append(float(truth[int(p[0]), int(p[1])]))
    return Run(method, [int(i) for i in idx], [float(v) for v in y], trace)


def analyse(settings, kind="terminal"):
    runs = {m: run_one(kind, m, settings, settings.n) for m in C.METHODS_BY_KIND[kind]}
    xs, ys = axes(kind)
    a = Analysis(kind, settings, runs, truth_grid(kind), optimum(kind), xs, ys)
    if kind == "terminal":
        a.formula, a.rule = formula_choice(), rule_choice()
    return a


def regret_pct(cost, opt_cost):
    """Mehrkosten gegen das Optimum in Prozent."""
    return 100.0 * (cost - opt_cost) / opt_cost


def final_costs(a):
    return {m: r.truth_trace[-1] for m, r in a.runs.items()}


# --- Vorgerechnete Studien -----------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def study():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_study.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _curves(raw):
    out = {}
    for m in raw:
        arr = np.array(raw[m])             # Läufe x Budget
        out[m] = {"median": np.median(arr, axis=0), "q25": np.quantile(arr, 0.25, axis=0), "q75": np.quantile(arr, 0.75, axis=0)}
    return out


def slice_curves(trucks):
    """Median und 25-/75-%-Quantil der wahren Kosten der Empfehlung je Budget und Verfahren (Scheibe, 20 feste Läufe)."""
    return _curves(study()["slice"][str(trucks)])


def branin_curves(noise):
    """Dasselbe für Branin (wahrer Wert der Empfehlung; das Minimum liegt bei 0,398)."""
    return _curves(study()["branin"][str(float(noise))])


def calibration_summary():
    cal = study()["calibration"]
    return {int(n): {"coverage_median": float(np.median(v["coverage"])), "coverage_q10": float(np.quantile(v["coverage"], 0.1)),
                     "mae": float(np.mean(v["mae"]))} for n, v in cal.items()}


def full_summary():
    """Gesamtkosten der Empfehlung in langer Simulation je Verfahren und Budget (volles Netz): Median, Mittel, 90-%-Quantil; dazu die Formeln."""
    st = study()
    out = {}
    for m in C.METHODS:
        out[m] = {}
        for b in C.STUDY_BUDGETS:
            v = np.array(st["full"][m][str(b)])
            out[m][b] = {"median": float(np.median(v)), "mean": float(v.mean()), "q90": float(np.quantile(v, 0.9))}
    return out, st["formula"], st["formula_rule"]


def bench_summary():
    """Glatte Testlandschaft ohne Strukturwissen (Hartmann6 auf 6 Eingaben): Median und 90-%-Quantil des Regrets je Verfahren und Budget."""
    st = study()["bench"]
    out = {}
    for m, per in st["regret"].items():
        out[m] = {int(b): {"median": float(np.median(v)), "q90": float(np.quantile(v, 0.9))} for b, v in per.items()}
    return out, st["optimum"]
