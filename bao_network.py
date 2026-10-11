"""Kopie des Simulators aus jackson-network-demo (Warteschlangentheorie, Stück 12): der "teure Simulator" dieser Demo.

Ereignisgesteuerte Simulation eines offenen Bediennetzes (Gate → Kran → Stapel mit Rückläufern, optional Platzgrenzen mit Blockieren).

Jede Station hat c gleiche Geräte, eine FIFO-Schlange und eine Dauer mit Mittel und Streuung (Variationskoeffizient² scv: 0 fest, 1/k Erlang-k, 1 exponentiell, über 1 Hyperexponential mit gleichen
Phasenanteilen am Mittel). Poisson-Ankünfte kommen an Station 0; nach der Abfertigung geht ein Lkw nach der Routing-Matrix P weiter oder verlässt das Netz.

**Blockieren nach Abfertigung.** Hat Station j eine Platzgrenze cap[j] (wartende + in Arbeit), und ein Lkw ist an Station i fertig, aber j ist voll, dann bleibt er an i stehen und hält sein Gerät besetzt,
bis in j ein Platz frei wird. Mit Platzgrenzen und Rückkopplung kann das Netz verklemmen; deshalb sind Platzgrenzen nur für Netze ohne Rückläufer erlaubt.

Zufall nur über übergebene `SplitMix64`-Ströme (Ankunft, je Station die Dauer, Routing), damit ein Lauf reproduzierbar ist und sich Strecken des Pfades zwischen Varianten teilen."""

import heapq
import math
from collections import deque
from dataclasses import dataclass

_MASK = (1 << 64) - 1


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def uniform(self):
        """Gleichverteilt auf [0, 1) mit 53 Bit."""
        return (self.next() >> 11) * (1.0 / (1 << 53))

    def expovariate(self, rate):
        """Exponentiell mit Mittel 1/rate (Inversion; 1 − u liegt in (0, 1], der Logarithmus ist endlich)."""
        return -math.log(1.0 - self.uniform()) / rate


def streams(seed, n_stations):
    """Die Zufallsströme eines Laufs: Ankunft, je Station die Dauer, Routing."""
    return SplitMix64(seed), [SplitMix64(seed + 1_000 * (i + 1) + 99_991) for i in range(n_stations)], SplitMix64(seed + 777_777)


def balanced_h2_probability(scv):
    """Anteil p₁ der ersten Phase der Hyperexponentialverteilung mit gleichen Phasenanteilen am Mittel (scv ≥ 1)."""
    return 0.5 * (1.0 + math.sqrt((scv - 1.0) / (scv + 1.0)))


def make_sampler(mean, scv, rng):
    """Funktion ohne Argument, die Werte mit dem Mittel `mean` und dem Variationskoeffizienten² `scv` zieht: fest, Erlang-k, exponentiell oder Hyperexponential."""
    if scv == 0:
        return lambda: mean
    if scv == 1:
        return lambda: rng.expovariate(1.0 / mean)
    if scv < 1:
        k = round(1.0 / scv)
        return lambda: sum(rng.expovariate(k / mean) for _ in range(k))
    p1 = balanced_h2_probability(scv)
    rate1, rate2 = 2.0 * p1 / mean, 2.0 * (1.0 - p1) / mean

    def draw():
        return rng.expovariate(rate1) if rng.uniform() < p1 else rng.expovariate(rate2)
    return draw


@dataclass
class SimResult:
    L: list                      # mittlere Zahl im System je Station (wartend, in Arbeit, blockiert), ohne die Einschwingphase
    mean_sojourn: float          # mittlere Gesamtzeit eines Lkw (Minuten), ohne die ersten warm_fraction der Abgänge
    n: int                       # ausgewertete Lkw
    rejected: int                # an einer vollen Station 0 abgewiesene Lkw
    end_time: float
    departures: list             # Zeitpunkte der Abfertigungsenden an `record_station` (leer, wenn nicht aufgezeichnet)


def simulate(gamma, c, mean, scv, p, trucks, seed, cap=None, warm_fraction=0.05, record_station=None, arrival=None, services=None, rng_route=None):
    """Simuliert `trucks` Ankünfte an Station 0 (Rate gamma je Minute) bis alle abgefertigt oder abgewiesen sind.
    Zum Testen lassen sich `arrival` (Funktion: Zwischenankunftszeit), `services` (Liste von Funktionen je Station) und `rng_route` (Objekt mit uniform()) von außen vorgeben."""
    n = len(c)
    scv = [float(scv)] * n if isinstance(scv, (int, float)) else list(scv)
    cap = [None] * n if cap is None else list(cap)
    if any(x is not None for x in cap) and any(p[i][j] > 0 for i in range(n) for j in range(n) if j <= i):
        raise ValueError("Platzgrenzen sind nur für Netze ohne Rückläufer erlaubt (Verklemmungsgefahr)")
    arr_rng, svc_rngs, route_rng = streams(seed, n)
    if arrival is None:
        arrival = lambda: arr_rng.expovariate(gamma)
    if services is None:
        services = [make_sampler(mean[i], scv[i], svc_rngs[i]) for i in range(n)]
    route_rng = rng_route or route_rng
    cum = []
    for i in range(n):
        acc, row = 0.0, []
        for j in range(n):
            acc += p[i][j]
            row.append(acc)
        cum.append(row)

    def route(i):
        u = route_rng.uniform()
        for j in range(n):
            if u < cum[i][j]:
                return j
        return None

    queue = [deque() for _ in range(n)]
    busy = [0] * n
    count = [0] * n
    blocked = [deque() for _ in range(n)]
    area = [0.0] * n
    ev = []
    seq = 0
    entry = {}
    sojourn = []
    departures = []
    warm = int(warm_fraction * trucks)
    t = last_t = t_start = 0.0
    finished = rejected = next_job = 0
    arrivals_left = trucks

    def has_room(j):
        return cap[j] is None or count[j] < cap[j]

    def try_start(i, now):
        nonlocal seq
        while busy[i] < c[i] and queue[i]:
            job = queue[i].popleft()
            busy[i] += 1
            seq += 1
            heapq.heappush(ev, (now + services[i](), seq, 1, i, job))

    def admit(j, job, now):
        count[j] += 1
        queue[j].append(job)
        try_start(j, now)

    seq += 1
    heapq.heappush(ev, (arrival(), seq, 0, -1, -1))
    while ev and finished + rejected < trucks:
        t, _, kind, i, job = heapq.heappop(ev)
        for k in range(n):
            area[k] += count[k] * (t - last_t)
        last_t = t
        if kind == 0:
            arrivals_left -= 1
            jid = next_job
            next_job += 1
            if has_room(0):
                entry[jid] = t
                admit(0, jid, t)
            else:
                rejected += 1
            if arrivals_left > 0:
                seq += 1
                heapq.heappush(ev, (t + arrival(), seq, 0, -1, -1))
        else:
            if record_station == i:
                departures.append(t)
            dest = route(i)
            if dest is None:
                busy[i] -= 1
                count[i] -= 1
                sojourn.append(t - entry.pop(job))
                finished += 1
                if finished == warm and warm > 0:
                    area = [0.0] * n
                    t_start = t
                try_start(i, t)
            elif has_room(dest):
                busy[i] -= 1
                count[i] -= 1
                admit(dest, job, t)
                try_start(i, t)
            else:
                blocked[i].append((job, dest))
        progress = any(blocked)
        while progress:
            progress = False
            for i2 in range(n):
                if blocked[i2] and has_room(blocked[i2][0][1]):
                    job2, dest2 = blocked[i2].popleft()
                    busy[i2] -= 1
                    count[i2] -= 1
                    admit(dest2, job2, t)
                    try_start(i2, t)
                    progress = True
    span = last_t - t_start
    kept = sojourn[warm:]
    return SimResult(L=[a / span for a in area], mean_sojourn=sum(kept) / len(kept), n=len(kept), rejected=rejected, end_time=last_t, departures=departures)
