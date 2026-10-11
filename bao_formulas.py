"""Kopie der Formeln aus jackson-network-demo (Stück 12), hier nur für die Referenz „Formel ohne Platzgrenze“ und die Orakel-Tests.

Formeln offener Bediennetze: Verkehrsgleichungen, Jackson-Netz (je Station M/M/c), Zerlegungsnäherung für beliebige Streuung (Whitt, Allen-Cunneen).

**Jackson-Netz.** Poisson-Ankünfte γ_j von außen, Station j hat c_j gleiche Geräte mit exponentieller Dauer (Mittel m_j), ein Lkw geht nach Station j mit Wahrscheinlichkeit P[i][j] von i zu j, sonst
verlässt er das Netz. Die Verkehrsgleichungen λ = γ + Pᵀλ liefern die Raten λ_j; ist jede Station stabil (ρ_j = λ_j m_j / c_j < 1), dann ist die gemeinsame Verteilung ein **Produkt** von M/M/c-
Verteilungen, jede Station verhält sich wie ein M/M/c mit der Rate λ_j. Die Gesamtzeit folgt aus Little: W = ΣL_j / Σγ_j.

**Zerlegung (QNA).** Bei nicht exponentieller Dauer gilt die Produktform nicht mehr; die Näherung rechnet jede Station als G/G/c mit Allen-Cunneen (Wq ≈ Erlang-C-Wartezeit · (ca² + cs²)/2) und reicht
die Streuung des Abgangsprozesses weiter: cd² = 1 + (1 − ρ²)(ca² − 1) + ρ²(cs² − 1)/√c (Whitt). Verzweigung: c² = 1 + p(cd² − 1) je Weg; Zusammenfluss: Mittel der c² gewichtet mit den Raten."""

import math

import numpy as np


def erlang_c_wait(c, a):
    """Mittlere Wartezeit eines M/M/c in Abfertigungsdauern (Dauer 1) bei Angebot a = λ·m (Erlang); stabil über die Erlang-B-Rekursion."""
    if a >= c:
        return math.inf
    b = 1.0
    for j in range(1, c + 1):
        b = a * b / (j + a * b)
    rho = a / c
    pw = b / (1.0 - rho * (1.0 - b))
    return pw / (c * (1.0 - rho))


def mmc_pmf(c, a, size):
    """Verteilung der Zahl im System eines M/M/c (Auslastung a/c < 1) für n = 0 … size − 1 (ohne Normierung auf den abgeschnittenen Bereich)."""
    rho = a / c
    terms = [1.0]
    for n in range(1, c + 1):
        terms.append(terms[-1] * a / n)
    p0 = 1.0 / (sum(terms[:c]) + terms[c] / (1.0 - rho))
    out = []
    for n in range(size):
        out.append(p0 * (terms[n] if n <= c else terms[c] * rho ** (n - c)))
    return np.array(out)


def traffic_rates(gamma, p):
    """Verkehrsgleichungen λ = γ + Pᵀλ: Ankunftsrate je Station (alle Lkw, auch Rückläufer)."""
    p = np.asarray(p, float)
    return np.linalg.solve(np.eye(len(p)) - p.T, np.asarray(gamma, float))


def station_metrics(c, mean, lam):
    """Kennzahlen einer M/M/c-Station bei Rate λ (je Minute), mittlere Dauer `mean` (Minuten): Auslastung, mittlere Wartezeit, Verweilzeit, mittlere Zahl im System."""
    a = lam * mean
    rho = a / c
    if rho >= 1.0:
        return {"rho": rho, "wq": math.inf, "w": math.inf, "L": math.inf}
    wq = erlang_c_wait(c, a) * mean
    return {"rho": rho, "wq": wq, "w": wq + mean, "L": lam * (wq + mean)}


def jackson(gamma, p, c, mean):
    """Jackson-Netz: Raten, Auslastungen, Wartezeiten, mittlere Zahl je Station und Gesamtzeit eines Lkw (Minuten, Little)."""
    lam = traffic_rates(gamma, p)
    st = [station_metrics(c[i], mean[i], lam[i]) for i in range(len(c))]
    L = np.array([s["L"] for s in st])
    total = float(L.sum()) / float(np.sum(gamma))
    return {"lam": lam, "rho": np.array([s["rho"] for s in st]), "wq": np.array([s["wq"] for s in st]), "w": np.array([s["w"] for s in st]), "L": L, "total": total,
            "stable": bool(all(s["rho"] < 1.0 for s in st))}


def bottleneck(rho):
    """Index der Station mit der höchsten Auslastung."""
    return int(np.argmax(rho))


def saturation_gamma(gamma, p, c, mean):
    """Äußere Ankunftsrate (Vielfaches von `gamma`), bei der die erste Station voll ausgelastet ist: Raten wachsen linear, also γ / max ρ."""
    lam = traffic_rates(gamma, p)
    rho = np.array([lam[i] * mean[i] / c[i] for i in range(len(c))])
    return float(np.sum(gamma)) / float(rho.max())


def departure_scv(rho, c, ca2, cs2):
    """Streuung (Variationskoeffizient²) des Abgangsprozesses einer Station nach Whitt: 1 + (1 − ρ²)(ca² − 1) + ρ²(cs² − 1)/√c."""
    return 1.0 + (1.0 - rho ** 2) * (ca2 - 1.0) + rho ** 2 * (cs2 - 1.0) / math.sqrt(c)


def allen_cunneen_wait(c, mean, lam, ca2, cs2):
    """Mittlere Wartezeit (Minuten) einer G/G/c-Station nach Allen-Cunneen: Erlang-C-Wartezeit · (ca² + cs²)/2."""
    return erlang_c_wait(c, lam * mean) * mean * (ca2 + cs2) / 2.0


def qna(gamma, p, c, mean, cs2, tol=1e-12, max_iter=10_000):
    """Zerlegungsnäherung (Whitt, QNA) für Netze mit Poisson-Ankünften von außen und beliebiger Streuung `cs2` (Zahl oder Liste je Station).
    Löst die Streuung der Ankunftsströme ca² je Station per Fixpunktiteration (Verzweigung und Zusammenfluss wie im Modul-Text)."""
    p = np.asarray(p, float)
    n = len(c)
    cs2 = np.full(n, float(cs2)) if np.isscalar(cs2) else np.asarray(cs2, float)
    lam = traffic_rates(gamma, p)
    gamma = np.asarray(gamma, float)
    rho = np.array([lam[i] * mean[i] / c[i] for i in range(n)])
    if (rho >= 1.0).any():
        inf = np.full(n, math.inf)
        return {"lam": lam, "rho": rho, "ca2": np.full(n, math.nan), "cd2": np.full(n, math.nan), "wq": inf, "L": inf, "total": math.inf, "stable": False}
    ca2 = np.ones(n)
    for _ in range(max_iter):
        cd2 = np.array([departure_scv(rho[i], c[i], ca2[i], cs2[i]) for i in range(n)])
        new = np.ones(n)
        for j in range(n):
            flow = gamma[j] * 1.0
            for i in range(n):
                if p[i, j] > 0:
                    flow += lam[i] * p[i, j] * (1.0 + p[i, j] * (cd2[i] - 1.0))
            new[j] = flow / lam[j]
        done = float(np.abs(new - ca2).max()) < tol
        ca2 = new
        if done:
            break
    cd2 = np.array([departure_scv(rho[i], c[i], ca2[i], cs2[i]) for i in range(n)])
    wq = np.array([allen_cunneen_wait(c[i], mean[i], lam[i], ca2[i], cs2[i]) for i in range(n)])
    L = np.array([lam[i] * (wq[i] + mean[i]) for i in range(n)])
    return {"lam": lam, "rho": rho, "ca2": ca2, "cd2": cd2, "wq": wq, "L": L, "total": float(L.sum()) / float(gamma.sum()), "stable": True}


def routing_matrix(r):
    """Routing des Terminals: Gate → Kran → Stapel; vom Stapel gehen r der Lkw zurück zum Kran (Umstapeln), die übrigen verlassen das Netz."""
    return [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, float(r), 0.0]]
