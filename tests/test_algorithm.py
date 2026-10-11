"""Einheiten des Kerns an Mini-Instanzen (Handrechnung): Kern, Vorhersage, erwartete Verbesserung, Suchverfahren."""

import math

import numpy as np
import pytest

import bao_algorithm as A


def test_matern_kernel_by_hand():
    """k(x, x) = Signalvarianz; bei r = 1/sqrt(5) ist sqrt(5) r = 1 und k = (1 + 1 + 1/3) e^-1 = 0.85836 (Varianz 1)."""
    a = np.array([[0.0]])
    r = 1.0 / math.sqrt(5.0)
    b = np.array([[r]])
    assert A.matern52(a, a, np.array([1.0]), 2.5)[0, 0] == pytest.approx(2.5)
    assert A.matern52(a, b, np.array([1.0]), 1.0)[0, 0] == pytest.approx((7.0 / 3.0) * math.exp(-1.0), rel=1e-12)


def test_matern_kernel_uses_the_lengthscale_per_input():
    """Halbe Längenmaß in Eingabe 0 wirkt wie doppelter Abstand in Eingabe 0 und lässt Eingabe 1 unberührt."""
    a, b = np.array([[0.0, 0.0]]), np.array([[0.2, 0.3]])
    k1 = A.matern52(a, b, np.array([0.5, 1.0]), 1.0)[0, 0]
    k2 = A.matern52(a, np.array([[0.4, 0.3]]), np.array([1.0, 1.0]), 1.0)[0, 0]
    assert k1 == pytest.approx(k2, rel=1e-12)


def test_expected_improvement_by_hand():
    """mu = 0, s = 1, bester Wert 0: EI = phi(0) = 0.39894; mu = 1: (-1) Phi(-1) + phi(-1) = 0.083316; ohne Unsicherheit: max(best - mu, 0)."""
    assert A.expected_improvement(np.array([0.0]), np.array([1.0]), 0.0)[0] == pytest.approx(0.3989422804, rel=1e-9)
    assert A.expected_improvement(np.array([1.0]), np.array([1.0]), 0.0)[0] == pytest.approx(0.0833155, rel=1e-5)
    assert A.expected_improvement(np.array([-2.0]), np.array([0.0]), 0.0)[0] == pytest.approx(2.0, abs=1e-9)
    assert A.expected_improvement(np.array([2.0]), np.array([0.0]), 0.0)[0] == pytest.approx(0.0, abs=1e-9)


def test_expected_improvement_grows_with_uncertainty_and_with_a_lower_mean():
    mu, best = 0.5, 0.0
    sds = np.array([0.1, 0.5, 1.0, 2.0])
    ei_sd = A.expected_improvement(np.full(4, mu), sds, best)
    assert np.all(np.diff(ei_sd) > 0)
    ei_mu = A.expected_improvement(np.array([1.0, 0.5, 0.0, -0.5]), np.full(4, 0.5), best)
    assert np.all(np.diff(ei_mu) > 0)


def test_gp_interpolates_and_is_certain_at_noise_free_observations(toy):
    cand, f = toy
    idx = [0, 6, 12, 20]
    gp = A.GaussianProcess(noise=1e-8).fit(cand[idx], f[idx])
    mu, sd = gp.predict(cand[idx])
    assert np.allclose(mu, f[idx], atol=1e-3)
    assert np.all(sd < 1e-2 * (f.max() - f.min()))
    _, sd_all = gp.predict(cand)
    assert sd_all[3] > sd_all[6]          # zwischen Beobachtungen unsicherer als an einer Beobachtung


def test_random_search_has_no_repeats_and_respects_the_budget(toy):
    cand, f = toy
    idx, y = A.random_search(len(cand), lambda i: f[i], 8, np.random.default_rng(1))
    assert len(idx) == 8 == len(set(idx)) and np.allclose(y, f[idx])
    idx_all, _ = A.random_search(len(cand), lambda i: f[i], 99, np.random.default_rng(1))
    assert len(idx_all) == len(cand)


def test_bo_never_repeats_a_candidate_and_uses_the_start_design(toy):
    cand, f = toy
    idx, y = A.bo_search(cand, lambda i: f[i], 9, np.random.default_rng(3), n_start=4, noise=1e-8)
    assert len(idx) == 9 == len(set(idx))
    assert np.allclose(y, f[idx])


def test_bo_finds_the_optimum_of_a_smooth_1d_function(toy):
    cand, f = toy
    for seed in range(5):
        idx, y = A.bo_search(cand, lambda i: f[i], 10, np.random.default_rng(seed), n_start=3, noise=1e-8)
        assert abs(A.recommend(idx, y) - 6) <= 1, seed


def test_bo_log_requires_positive_values_and_matches_on_a_positive_function(toy):
    cand, f = toy
    idx, y = A.bo_search(cand, lambda i: f[i], 10, np.random.default_rng(2), n_start=3, noise=1e-8, log=True)
    assert abs(A.recommend(idx, y) - 6) <= 1


def test_tpe_never_repeats_and_prefers_the_good_region(toy):
    cand, f = toy
    idx, y = A.tpe_search(cand, lambda i: f[i], 12, np.random.default_rng(4), n_start=4, bandwidth=0.15)
    assert len(idx) == 12 == len(set(idx))
    assert abs(A.recommend(idx, y) - 6) <= 2


def test_recommend_is_the_best_observation():
    assert A.recommend([5, 9, 2], [3.0, 1.0, 2.0]) == 9


def test_gp_state_returns_one_value_per_candidate(toy):
    cand, f = toy
    mu, sd, ei, gp = A.gp_state(cand, [0, 6, 20], f[[0, 6, 20]], noise=1e-8)
    assert mu.shape == sd.shape == ei.shape == (len(cand),)
    assert np.all(ei >= 0) and np.all(sd >= 0)
