"""Terminal als Zielfunktion und vorgerechnete Tabelle: Form, Handrechnung der Kosten, Ziehen ohne Zurücklegen, Wahrheit gegen Neurechnung."""

import numpy as np
import pytest

import bao_constants as C
import bao_scenario as S


def test_device_and_total_cost_by_hand():
    """Gate 4 x 20 = 80; Kran 4 x 120 x 0,7^2 = 235,2; Stapel 5 x 60 x 1^2 = 300; 5 Plätze x 4 = 20: zusammen 635,2. Wartekosten: 40 x 33 x 30/60 = 660."""
    assert S.device_cost(5, 1.0) == pytest.approx(635.2)
    assert S.total_cost(5, 1.0, 30.0) == pytest.approx(635.2 + 660.0)


def test_stack_means_scale_with_the_speed():
    m = S.stack_means(0.8)
    assert m[0] == 3.0 and m[1] == pytest.approx(2.4 / 0.7) and m[2] == pytest.approx(5.0)


def test_unstable_network_is_capped_without_simulating():
    """Tempo 0,6 bei 1 Stapelgerät wäre Überlast; die Funktion liefert die Kappung (Handrechnung: ρ = 0,55 x 6,67 / 1 > 1)."""
    from bao_constants import GAMMA_PER_HOUR
    assert GAMMA_PER_HOUR / 60 * (C.STAPEL_BASE_MEAN / 0.6) / 1 > 1
    assert S.full_time(4, 4, 1, 13, 0.7, 0.6, 1000, 1) == C.T_CAP


def test_table_has_the_grid_shape_and_replicates():
    t = S.table()
    assert t["truth"].shape == (C.N_PLACES, C.N_SPEEDS)
    for trucks in C.TRUCK_CHOICES:
        assert t["reps"][trucks].shape == (C.N_PLACES, C.N_SPEEDS, C.REPLICATES)


def test_observe_draws_without_replacement_within_the_replicates():
    term = S.Terminal(15000, seed=1)
    vals = [term.observe(3, 4) for _ in range(C.REPLICATES)]
    assert sorted(vals) == sorted(S.table()["reps"][15000][3, 4].tolist())
    assert term.calls == C.REPLICATES
    term.observe(3, 4)                         # danach beginnt der Vorrat von vorn, ohne Fehler


def test_observations_are_reproducible_for_equal_seeds():
    a = S.Terminal(3000, seed=9)
    b = S.Terminal(3000, seed=9)
    assert [a.observe(2, 2) for _ in range(5)] == [b.observe(2, 2) for _ in range(5)]


def test_replicate_means_match_the_long_run_truth_on_stable_cells():
    t = S.table()
    ok = t["truth"] < 2000
    for trucks, tol in ((15000, 0.01), (3000, 0.02)):
        mean = t["reps"][trucks].mean(-1)
        assert np.median(np.abs(mean - t["truth"])[ok] / t["truth"][ok]) < tol


def test_truth_cell_matches_a_fresh_long_simulation():
    """Unabhängige Neurechnung einer Zelle mit anderen Seeds: innerhalb 1,5 %."""
    i, j = 3, 4
    fresh = np.mean([S.simulate_cost(C.PLACES[i], C.SPEEDS[j], 100_000, 555_000 + r) for r in range(2)])
    assert fresh == pytest.approx(S.table()["truth"][i, j], rel=0.015)


def test_formula_ignores_the_places_except_for_their_cost():
    """Die Formel hängt von der Platzzahl nur über die Platzkosten ab: Differenz = 4 EUR/h je Platz."""
    assert S.formula_cost(7, 0.8) - S.formula_cost(6, 0.8) == pytest.approx(C.K_PLACE)


def test_full_pool_is_unique_and_on_the_grid():
    pool = S.full_pool()
    assert len(np.unique(pool, axis=0)) == len(pool) > 3500
    d = S.full_decode(pool[10])
    assert d[0] in C.FULL_GATE and d[3] in C.FULL_PLACES and d[4] in C.FULL_SPEED_KRAN
