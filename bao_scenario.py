"""Das Terminal als Zielfunktion: Kosten je Stunde aus Geräten, Stellplätzen und Wartezeit; Simulation und vorgerechnete Tabelle."""

import json
import os

import numpy as np

import bao_constants as C
import bao_formulas as F
import bao_network as N


def stack_means(speed):
    """Mittlere Dauer je Station (Gate, Kran, Stapel) bei Stapeltempo `speed`."""
    return [C.GATE_MEAN, C.KRAN_BASE_MEAN / C.KRAN_SPEED, C.STAPEL_BASE_MEAN / speed]


def device_cost(places, speed):
    """Gerätekosten (EUR/h): Gate, Kran (mit Tempo C.KRAN_SPEED), Stapel (mit Tempo `speed`) und Stellplätze. Tempo geht quadratisch ein."""
    return (C.K_GATE * C.GATE_SERVERS + C.K_KRAN * C.KRAN_SERVERS * C.KRAN_SPEED ** 2
            + C.K_STAPEL * C.STAPEL_SERVERS * speed ** 2 + C.K_PLACE * places)


def total_cost(places, speed, total_time):
    """Gesamtkosten (EUR/h) = Gerätekosten + Wartekosten: V_WAIT je Lkw-Stunde im System, γ Lkw je Stunde."""
    return device_cost(places, speed) + C.V_WAIT * C.GAMMA_PER_HOUR * total_time / 60.0


def simulate_time(places, speed, trucks, seed):
    """Eine Simulation: mittlere Gesamtzeit eines Lkw in Minuten, bei Überlast auf C.T_CAP gekappt."""
    m = stack_means(speed)
    servers = [C.GATE_SERVERS, C.KRAN_SERVERS, C.STAPEL_SERVERS]
    gamma = C.GAMMA_PER_HOUR / 60.0
    if max(gamma * m[i] / servers[i] for i in range(3)) >= 1.0:
        return C.T_CAP
    res = N.simulate(gamma, servers, m, C.CS2, F.routing_matrix(0.0), trucks, seed, cap=[None, None, places])
    return min(res.mean_sojourn, C.T_CAP)


def simulate_cost(places, speed, trucks, seed):
    return total_cost(places, speed, simulate_time(places, speed, trucks, seed))


def formula_time(places, speed):
    """Zerlegungsnäherung (Whitt) OHNE Platzgrenze: die Formel kennt `places` nicht (nur die Kosten)."""
    m = stack_means(speed)
    servers = [C.GATE_SERVERS, C.KRAN_SERVERS, C.STAPEL_SERVERS]
    g = np.array([C.GAMMA_PER_HOUR / 60.0, 0.0, 0.0])
    p = F.routing_matrix(0.0)
    if not F.jackson(g, p, servers, m)["stable"]:
        return C.T_CAP
    return min(F.qna(g, p, servers, m, [C.CS2] * 3)["total"], C.T_CAP)


def formula_cost(places, speed):
    return total_cost(places, speed, formula_time(places, speed))


_TABLE = None


def table():
    """Vorgerechnete Tabelle (generate_precomputed.py): Wahrheit und Wiederholungen je Gitterpunkt."""
    global _TABLE
    if _TABLE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_slice.json")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        _TABLE = {
            "truth": np.array(raw["truth"], float),
            "reps": {int(k): np.array(v, float) for k, v in raw["reps"].items()},
        }
    return _TABLE


class Terminal:
    """Zielfunktion auf dem 12 x 17-Gitter. `observe` liefert eine verrauschte Auswertung aus der Tabelle (echte Simulationen), `truth` die Wahrheit."""

    def __init__(self, trucks, seed):
        self.trucks = trucks
        self.rng = np.random.default_rng(seed)
        self.used = {}
        self.calls = 0

    def observe(self, i, j):
        """Verrauschte Auswertung für Platz-Index i, Tempo-Index j: eine noch nicht gezogene Wiederholung der Tabelle (ohne Zurücklegen je Punkt)."""
        reps = table()["reps"][self.trucks][i, j]
        used = self.used.setdefault((i, j), [])
        free = [r for r in range(len(reps)) if r not in used]
        if not free:
            used.clear()
            free = list(range(len(reps)))
        r = int(self.rng.choice(free))
        used.append(r)
        self.calls += 1
        return float(reps[r])

    @staticmethod
    def truth(i, j):
        return float(table()["truth"][i, j])


def full_means(s_kran, s_stapel):
    """Mittlere Dauer je Station im vollen Netz."""
    return [C.GATE_MEAN, C.KRAN_BASE_MEAN / s_kran, C.STAPEL_BASE_MEAN / s_stapel]


def full_device_cost(c_gate, c_kran, c_stapel, places, s_kran, s_stapel):
    """Gerätekosten (EUR/h) im vollen Netz: Geräte je Station, Tempo quadratisch, Stellplätze."""
    return C.K_GATE * c_gate + C.K_KRAN * c_kran * s_kran ** 2 + C.K_STAPEL * c_stapel * s_stapel ** 2 + C.K_PLACE * places


def full_time(c_gate, c_kran, c_stapel, places, s_kran, s_stapel, trucks, seed):
    """Mittlere Gesamtzeit (Minuten) im vollen Netz, bei Überlast gekappt."""
    m = full_means(s_kran, s_stapel)
    servers = [c_gate, c_kran, c_stapel]
    gamma = C.GAMMA_PER_HOUR / 60.0
    if max(gamma * m[i] / servers[i] for i in range(3)) >= 1.0:
        return C.T_CAP
    res = N.simulate(gamma, servers, m, C.CS2, F.routing_matrix(0.0), trucks, seed, cap=[None, None, places])
    return min(res.mean_sojourn, C.T_CAP)


def full_cost(c_gate, c_kran, c_stapel, places, s_kran, s_stapel, trucks, seed):
    """Gesamtkosten (EUR/h) einer Simulation im vollen Netz."""
    t = full_time(c_gate, c_kran, c_stapel, places, s_kran, s_stapel, trucks, seed)
    return full_device_cost(c_gate, c_kran, c_stapel, places, s_kran, s_stapel) + C.V_WAIT * C.GAMMA_PER_HOUR * t / 60.0


def full_formula_cost(c_gate, c_kran, c_stapel, places, s_kran, s_stapel):
    """Kosten nach der Formel ohne Platzgrenze (Zerlegung); instabile Netze auf die Kappung."""
    m = full_means(s_kran, s_stapel)
    servers = [c_gate, c_kran, c_stapel]
    g = np.array([C.GAMMA_PER_HOUR / 60.0, 0.0, 0.0])
    p = F.routing_matrix(0.0)
    t = C.T_CAP if not F.jackson(g, p, servers, m)["stable"] else min(F.qna(g, p, servers, m, [C.CS2] * 3)["total"], C.T_CAP)
    return full_device_cost(c_gate, c_kran, c_stapel, places, s_kran, s_stapel) + C.V_WAIT * C.GAMMA_PER_HOUR * t / 60.0


def full_decode(unit):
    """Einheitswürfel-Punkt (6 Koordinaten) auf Rasterwerte (c_gate, c_kran, c_stapel, places, s_kran, s_stapel)."""
    levels = [C.FULL_GATE, C.FULL_KRAN, C.FULL_STAPEL, C.FULL_PLACES, C.FULL_SPEED_KRAN, C.FULL_SPEED_STAPEL]
    return tuple(lv[int(round(float(u) * (len(lv) - 1)))] for u, lv in zip(unit, levels))


def full_pool(n=None, seed=1):
    """Kandidatenmenge der Studie: gescrambelte Sobol-Punkte auf das Raster gerastert (eindeutig, fester Seed)."""
    from scipy.stats import qmc
    pts = qmc.Sobol(6, scramble=True, seed=seed).random(n or C.FULL_POOL)
    snapped = np.column_stack([np.round(pts[:, j] * (L - 1)) / (L - 1) for j, L in enumerate(C.FULL_LEVELS)])
    return np.unique(snapped, axis=0)


def grid_points(shape=None):
    """Alle Gitterpunkte als Array (Zeilen-Index, Spalten-Index); Standard: die Terminal-Scheibe."""
    rows, cols = shape or (C.N_PLACES, C.N_SPEEDS)
    return np.array([(i, j) for i in range(rows) for j in range(cols)], dtype=int)


def to_unit(pts, shape=None):
    """Gitterindizes auf den Einheitswürfel (für das Ersatzmodell)."""
    rows, cols = shape or (C.N_PLACES, C.N_SPEEDS)
    pts = np.asarray(pts, float)
    return np.column_stack([pts[:, 0] / (rows - 1), pts[:, 1] / (cols - 1)])


class BraninField:
    """Branin auf einem 31 x 31-Gitter mit Gauß-Rauschen. Zeilen = x2, Spalten = x1 (wie die Terminal-Scheibe: Zeile, Spalte)."""

    _GRID = None

    def __init__(self, noise, seed):
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed + 31)
        self.calls = 0

    @classmethod
    def _grid(cls):
        if cls._GRID is None:
            from bao_benchmarks import branin_grid
            cls._GRID = branin_grid(C.BRANIN_STEPS)
        return cls._GRID

    @classmethod
    def axes(cls):
        x1, x2, truth = cls._grid()
        return x1, x2, truth

    @classmethod
    def truth_grid(cls):
        return cls._grid()[2]

    def observe(self, i, j):
        """Wahrer Wert plus Gauß-Rauschen mit der gewählten Standardabweichung."""
        self.calls += 1
        return float(self.truth_grid()[i, j] + self.noise * self.rng.standard_normal())
