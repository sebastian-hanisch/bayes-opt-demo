"""Orakel-Tests bayes-opt-demo: Kern gegen unabhängige Rechenwege (ORACLE-PLAYBOOK).

* Gauß-Prozess: Vorhersage bei den gelernten Hyperparametern gegen scikit-learn (anderer Code, gleiche Formeln).
* Erwartete Verbesserung: geschlossene Form gegen numerische Integration (scipy.integrate.quad) und gegen Monte-Carlo.
* Simulator: M/M/1 gegen die geschlossene Formel, Jackson-Netz (exponentiell) gegen die Formel aus bao_formulas, Platzgrenze gegen Handrechnung.
* Suche: Brute-Force-Aufzählung kleiner Aufgaben (BO findet das Optimum, wo es aufzählbar ist).
"""

import numpy as np
import pytest

import bao_algorithm as A
import bao_constants as C
import bao_formulas as F
import bao_network as N


# --- Gauß-Prozess gegen scikit-learn -------------------------------------------------------------------------------------------------------

def test_gp_prediction_matches_scikit_learn_at_the_learned_hyperparameters():
    gp_mod = pytest.importorskip("sklearn.gaussian_process")
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    rng = np.random.default_rng(11)
    for trial in range(5):
        X = rng.random((10, 2))
        y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.standard_normal(10)
        gp = A.GaussianProcess().fit(X, y)
        z = (y - gp.y_mean) / gp.y_std
        kern = ConstantKernel(gp.variance, "fixed") * Matern(length_scale=gp.lengthscales, nu=2.5, length_scale_bounds="fixed")
        sk = gp_mod.GaussianProcessRegressor(kern, alpha=gp.noise + 1e-8, optimizer=None).fit(X, z)
        Xs = rng.random((15, 2))
        m_ref, s_ref = sk.predict(Xs, return_std=True)
        mu, sd = gp.predict(Xs)
        assert np.allclose((mu - gp.y_mean) / gp.y_std, m_ref, atol=1e-6), trial
        assert np.allclose(sd / gp.y_std, s_ref, atol=1e-5), trial


def test_learned_hyperparameters_reach_a_likelihood_at_least_as_high_as_a_default_start():
    """Die Randwahrscheinlichkeit nach dem Lernen ist nicht schlechter als bei festen Standardwerten (Plausibilität der Optimierung)."""
    rng = np.random.default_rng(5)
    X = rng.random((12, 2))
    y = np.sin(4 * X[:, 0]) + 0.3 * X[:, 1] + 0.05 * rng.standard_normal(12)
    gp = A.GaussianProcess().fit(X, y)
    z = (y - gp.y_mean) / gp.y_std
    th_learned = np.concatenate([np.log(gp.lengthscales), [np.log(gp.variance)], [np.log(gp.noise)]])
    th_default = np.concatenate([np.log([0.5, 0.5]), [0.0], [np.log(0.1)]])
    assert gp._neg_log_marginal(th_learned, X, z) <= gp._neg_log_marginal(th_default, X, z) + 1e-9


# --- Erwartete Verbesserung gegen Integration ------------------------------------------------------------------------------------------------

def test_expected_improvement_matches_numerical_integration_and_monte_carlo():
    quad = pytest.importorskip("scipy.integrate").quad
    from scipy.stats import norm

    rng = np.random.default_rng(3)
    for mu, sd, best in [(0.3, 0.5, 0.0), (-0.2, 0.1, 0.0), (1.0, 2.0, 0.0), (0.0, 1.0, 0.5)]:
        ref, _ = quad(lambda f: max(best - f, 0.0) * norm.pdf(f, mu, sd), mu - 12 * sd, mu + 12 * sd, limit=200)
        assert A.expected_improvement(np.array([mu]), np.array([sd]), best)[0] == pytest.approx(ref, rel=1e-6, abs=1e-9)
        mc = np.maximum(best - (mu + sd * rng.standard_normal(400_000)), 0.0).mean()
        assert A.expected_improvement(np.array([mu]), np.array([sd]), best)[0] == pytest.approx(mc, rel=2e-2, abs=1e-3)


# --- Simulator gegen Formeln --------------------------------------------------------------------------------------------------------------

def test_single_server_queue_matches_the_mm1_formula():
    """M/M/1, λ = 0,5, Dauer 1: mittlere Verweilzeit 1 / (μ − λ) = 2."""
    res = N.simulate(0.5, [1], [1.0], 1.0, [[0.0]], 120_000, seed=3)
    assert res.mean_sojourn == pytest.approx(2.0, rel=0.04)


def test_exponential_tandem_matches_the_jackson_formula():
    p = F.routing_matrix(0.0)
    c, mean, gamma = [3, 2, 4], [3.0, 2.4, 4.0], 33.0 / 60.0
    ref = F.jackson(np.array([gamma, 0.0, 0.0]), p, c, mean)["total"]
    sims = [N.simulate(gamma, c, mean, 1.0, p, 60_000, seed=s).mean_sojourn for s in (1, 2, 3)]
    assert np.mean(sims) == pytest.approx(ref, rel=0.03)


def test_a_huge_place_limit_changes_nothing_and_a_tiny_one_blocks():
    p = F.routing_matrix(0.0)
    args = (33.0 / 60.0, [4, 4, 5], [3.0, 2.4 / 0.7, 4.0 / 0.8], C.CS2, p, 8000, 7)
    free = N.simulate(*args).mean_sojourn
    assert N.simulate(*args, cap=[None, None, 10_000]).mean_sojourn == free      # gleiche Zufallsströme: identisch
    assert N.simulate(*args, cap=[None, None, 2]).mean_sojourn > 3 * free        # zwei Plätze bei fünf Geräten: der Kran blockiert


def test_blocking_by_hand_two_places_one_server_each():
    """Handrechnung (deterministisch): Station 0 (1 Gerät, Dauer 1) füttert Station 1 (1 Gerät, Dauer 3, Platzgrenze 1). Ankunft alle 2 Minuten.
    Lkw 0: 0..1 Station 0, 1..4 Station 1. Lkw 1 kommt bei 2, fertig bei 3, Platz an Station 1 belegt bis 4: wartet blockiert bis 4, 4..7."""
    t = {"i": 0}

    def arrival():
        t["i"] += 1
        return 0.0 if t["i"] == 1 else 2.0

    class Route:
        def uniform(self):
            return 0.0

    res = N.simulate(1.0, [1, 1], [1.0, 3.0], 0, [[0.0, 1.0], [0.0, 0.0]], 2, 1, cap=[None, 1], warm_fraction=0.0,
                     arrival=arrival, services=[lambda: 1.0, lambda: 3.0], rng_route=Route())
    # Lkw 0: Ankunft 0, Abgang 4 (Verweilzeit 4); Lkw 1: Ankunft 2, Abgang 7 (Verweilzeit 5); Mittel 4,5
    assert res.mean_sojourn == pytest.approx(4.5)


# --- Suche gegen Aufzählung ------------------------------------------------------------------------------------------------------------------

def test_bo_reaches_the_enumerated_optimum_on_small_two_dimensional_grids():
    """Aufzählung aller 49 Punkte eines 7 x 7-Gitters liefert das Optimum; BO mit 14 Auswertungen findet es oder ist höchstens ein Gitterschritt entfernt."""
    g = np.linspace(0, 1, 7)
    cand = np.array([[a, b] for a in g for b in g])
    for shift in [(0.3, 0.7), (0.6, 0.2), (0.5, 0.5), (0.2, 0.2)]:
        f = np.array([(x - shift[0]) ** 2 + 2 * (y - shift[1]) ** 2 + 0.3 * np.sin(6 * x) for x, y in cand])
        opt = int(np.argmin(f))
        hits = 0
        for seed in range(6):
            idx, y = A.bo_search(cand, lambda i: f[i], 14, np.random.default_rng(seed), n_start=4, noise=1e-8)
            best = A.recommend(idx, y)
            hits += int(np.abs(cand[best] - cand[opt]).max() <= 1 / 6 + 1e-9)
        assert hits >= 5, shift
