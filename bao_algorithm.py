"""Kern der Demo: Gauß-Prozess (Matérn-5/2, je Eingabe ein Längenmaß), Expected Improvement, Suchverfahren auf einer endlichen Kandidatenmenge.

Alles auf dem Einheitswürfel, Minimierung, nur numpy/scipy. Eine Einheit = eine Regel (DEMO-PLAYBOOK §1b): `matern52` (Kern),
`GaussianProcess.fit/predict` (Lernen, Vorhersage), `expected_improvement` (Akquisition), `random_search`, `tpe_search`, `bo_search` (Auswahlregeln).
Zufall nur über übergebene `numpy.random.Generator`.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm


def matern52(A, B, lengthscales, variance):
    """Matérn-5/2-Kern k(a, b) = σ² (1 + √5 r + 5r²/3) exp(−√5 r), r = Abstand mit je Eingabe eigenem Längenmaß."""
    d = np.sqrt(np.maximum((((A[:, None, :] - B[None, :, :]) / lengthscales) ** 2).sum(-1), 0.0)) * np.sqrt(5.0)
    return variance * (1.0 + d + d * d / 3.0) * np.exp(-d)


class GaussianProcess:
    """Gauß-Prozess-Regression mit Matérn-5/2-Kern. Ausgabe wird standardisiert; Längenmaße, Signalvarianz und (optional) Rauschvarianz
    werden durch Maximieren der Randwahrscheinlichkeit gelernt (L-BFGS-B, mehrere Startpunkte, feste Seeds)."""

    LS_BOUNDS = (0.05, 10.0)

    def __init__(self, noise=None, restarts=3):
        self.noise_fixed = noise
        self.restarts = restarts

    def _unpack(self, theta, d):
        ls = np.exp(theta[:d])
        var = np.exp(theta[d])
        nz = self.noise_fixed if self.noise_fixed is not None else np.exp(theta[d + 1])
        return ls, var, nz

    def _neg_log_marginal(self, theta, X, z):
        ls, var, nz = self._unpack(theta, X.shape[1])
        K = matern52(X, X, ls, var) + (nz + 1e-8) * np.eye(len(X))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(c, z)
        return 0.5 * z @ a + np.log(np.diag(c[0])).sum() + 0.5 * len(z) * np.log(2 * np.pi)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.X = X
        self.y_mean = float(y.mean())
        self.y_std = max(float(y.std()), 1e-9)
        z = (y - self.y_mean) / self.y_std
        d = X.shape[1]
        lo, hi = np.log(self.LS_BOUNDS[0]), np.log(self.LS_BOUNDS[1])
        bounds = [(lo, hi)] * d + [(np.log(0.05), np.log(10.0))]
        if self.noise_fixed is None:
            bounds.append((np.log(1e-6), np.log(2.0)))
        starts = np.random.default_rng(0).uniform(0.15, 1.0, (self.restarts, d))
        best = None
        for r in range(self.restarts):
            th0 = np.concatenate([np.log(starts[r]), [0.0]] + ([[np.log(1e-2)]] if self.noise_fixed is None else []))
            res = minimize(self._neg_log_marginal, th0, args=(X, z), method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.lengthscales, self.variance, self.noise = self._unpack(best.x, d)
        K = matern52(X, X, self.lengthscales, self.variance) + (self.noise + 1e-8) * np.eye(len(X))
        self._chol = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._chol, z)
        return self

    def predict(self, Xs):
        """Mittelwert und Standardabweichung der **Funktion** (ohne Beobachtungsrauschen) an den Stellen Xs."""
        Xs = np.asarray(Xs, float)
        Ks = matern52(Xs, self.X, self.lengthscales, self.variance)
        mu = Ks @ self._alpha
        v = cho_solve(self._chol, Ks.T)
        var = np.maximum(self.variance - (Ks * v.T).sum(1), 1e-12)
        return mu * self.y_std + self.y_mean, np.sqrt(var) * self.y_std


def expected_improvement(mu, sd, best):
    """Erwartete Verbesserung (Minimierung) gegenüber dem bisher besten Wert `best`: E[max(best − f, 0)], f ~ N(mu, sd²)."""
    sd = np.maximum(np.asarray(sd, float), 1e-12)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


# --- Suchverfahren auf einer endlichen Kandidatenmenge (Zeilen von `cand`, Einheitswürfel) -----------------------------------------------

def random_search(n_cand, evaluate, budget, rng):
    """Zufällig ohne Zurücklegen. `evaluate(i)` liefert die (verrauschte) Auswertung für Kandidat i. Gibt (Indizes, Werte) zurück."""
    idx = [int(i) for i in rng.choice(n_cand, size=min(budget, n_cand), replace=False)]
    return idx, [evaluate(i) for i in idx]


def _start_design(n_cand, n_start, rng):
    return [int(i) for i in rng.choice(n_cand, size=n_start, replace=False)]


def tpe_search(cand, evaluate, budget, rng, n_start=5, gamma=0.25, bandwidth=0.15):
    """TPE-Idee (Bergstra u. a.): die besten gamma der Beobachtungen bilden die Dichte l, die übrigen g; gewählt wird der noch nicht
    ausgewertete Kandidat mit dem größten l/g. Einfache Fassung mit Gauß-Kernen gleicher Breite (keine Mischung aus Prior und Kernen)."""
    n = len(cand)
    idx = _start_design(n, min(n_start, budget), rng)
    y = [evaluate(i) for i in idx]
    while len(idx) < budget:
        order = np.argsort(y)
        n_good = max(2, int(np.ceil(gamma * len(y))))
        good = cand[[idx[i] for i in order[:n_good]]]
        bad = cand[[idx[i] for i in order[n_good:]]]

        def log_density(pts):
            d2 = ((cand[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
            return np.log(np.exp(-0.5 * d2 / bandwidth ** 2).mean(1) + 1e-12)

        score = log_density(good) - (log_density(bad) if len(bad) else 0.0)
        score[idx] = -np.inf
        i = int(np.argmax(score))
        idx.append(i)
        y.append(evaluate(i))
    return idx, y


def bo_search(cand, evaluate, budget, rng, n_start=5, log=False, noise=None):
    """Bayessche Optimierung: Startdesign zufällig, danach je Schritt GP anpassen und den noch nicht ausgewerteten Kandidaten mit der größten
    erwarteten Verbesserung wählen. `log`: Gauß-Prozess auf log(Wert) (Werte müssen positiv sein). `noise`: feste Rauschvarianz oder None (lernen)."""
    n = len(cand)
    idx = _start_design(n, min(n_start, budget), rng)
    y = [evaluate(i) for i in idx]
    while len(idx) < budget:
        yt = np.log(y) if log else np.asarray(y, float)
        gp = GaussianProcess(noise=noise).fit(cand[idx], yt)
        mu, sd = gp.predict(cand)
        best = float(np.min(gp.predict(cand[idx])[0]))     # bester Mittelwert an den Beobachtungen: robuster gegen Rauschen als der beste Messwert
        ei = expected_improvement(mu, sd, best)
        ei[idx] = -np.inf
        i = int(np.argmax(ei))
        idx.append(i)
        y.append(evaluate(i))
    return idx, y


def gp_state(cand, idx, y, log=False, noise=None):
    """Zustand des Ersatzmodells nach den Beobachtungen (idx, y): Mittelwert, Standardabweichung (auf der Kostenskala, bei log: Rücktransformation
    des Mittelwerts) und erwartete Verbesserung je Kandidat. Für die Anschauung; die Suche selbst nutzt `bo_search`."""
    yt = np.log(y) if log else np.asarray(y, float)
    gp = GaussianProcess(noise=noise).fit(cand[idx], yt)
    mu, sd = gp.predict(cand)
    best = float(np.min(gp.predict(cand[idx])[0]))
    ei = expected_improvement(mu, sd, best)
    if log:
        return np.exp(mu), np.exp(mu) * sd, ei, gp
    return mu, sd, ei, gp


def recommend(idx, y):
    """Empfehlung: der Kandidat mit der besten (kleinsten) Beobachtung."""
    return idx[int(np.argmin(y))]
