"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (vorgerechnete Tabelle und Studien, feste Läufe).

Plattformrobust: Bänder statt exakter Werte (die Tabellen sind festgeschrieben; die Bänder decken eine Neuberechnung auf anderer Plattform ab)."""

import numpy as np
import pytest

import bao_constants as C
import bao_evaluation as ev
import bao_scenario as S


def med_regret(curves, method, n, opt):
    return float(curves[method]["median"][n - 1]) - opt


def test_branin_grid_minimum_is_close_to_the_literature_value():
    assert ev.optimum("branin")[2] == pytest.approx(0.427, abs=0.02)       # Gitter 31 x 31; Literaturwert 0,398 liegt zwischen den Gitterpunkten


def test_branin_sweep_without_noise_bo_and_tpe_reach_the_grid_optimum_random_does_not():
    cur = ev.branin_curves(0.0)
    opt = ev.optimum("branin")[2]
    assert med_regret(cur, "bo", 30, opt) == pytest.approx(0.0, abs=0.05)
    assert med_regret(cur, "tpe", 30, opt) == pytest.approx(0.0, abs=0.05)
    assert med_regret(cur, "random", 30, opt) == pytest.approx(0.71, abs=0.3)


def test_branin_sweep_with_moderate_noise_the_advantage_comes_late():
    cur = ev.branin_curves(5.0)
    opt = ev.optimum("branin")[2]
    # bei 16 Auswertungen noch kein klarer Vorsprung ...
    assert med_regret(cur, "random", 16, opt) == pytest.approx(3.66, abs=0.8)
    assert med_regret(cur, "bo", 16, opt) == pytest.approx(3.13, abs=0.8)
    assert med_regret(cur, "bo", 16, opt) > 0.6 * med_regret(cur, "random", 16, opt)
    # ... bei 30 schon
    assert med_regret(cur, "bo", 30, opt) == pytest.approx(0.94, abs=0.5)
    assert med_regret(cur, "random", 30, opt) == pytest.approx(3.09, abs=0.8)
    assert med_regret(cur, "tpe", 30, opt) == pytest.approx(1.56, abs=0.6)
    assert med_regret(cur, "bo", 30, opt) < 0.5 * med_regret(cur, "random", 30, opt)


def test_branin_sweep_with_strong_noise_random_search_stays_far_behind():
    cur = ev.branin_curves(20.0)
    opt = ev.optimum("branin")[2]
    assert med_regret(cur, "random", 30, opt) == pytest.approx(13.5, abs=3.0)
    assert med_regret(cur, "bo", 30, opt) == pytest.approx(5.2, abs=2.0)
    assert med_regret(cur, "tpe", 30, opt) == pytest.approx(5.1, abs=2.0)


def test_hartmann6_study():
    b, _ = ev.bench_summary()
    assert b["random"][40]["median"] == pytest.approx(1.45, abs=0.3)
    assert b["tpe"][40]["median"] == pytest.approx(0.56, abs=0.3)
    assert b["bo"][40]["median"] == pytest.approx(0.08, abs=0.1)
    assert b["random"][60]["median"] == pytest.approx(1.09, abs=0.3)
    assert b["tpe"][60]["median"] == pytest.approx(0.15, abs=0.15)
    assert b["bo"][20]["median"] < b["random"][20]["median"]


def test_terminal_slice_geometry():
    t = S.table()["truth"]
    assert t.shape == (12, 17)
    i, j = np.unravel_index(int(np.argmin(t)), t.shape)
    assert C.PLACES[i] == 5 and C.SPEEDS[j] == pytest.approx(0.7)
    assert t.min() == pytest.approx(790.6, abs=6.0)
    assert float((t <= t.min() * 1.05).mean()) == pytest.approx(0.216, abs=0.03)       # ein Fünftel aller Einstellungen liegt höchstens 5 % über dem Optimum
    assert float((t > 4000).mean()) == pytest.approx(0.078, abs=0.02)                  # Überlast: 16 von 204


def test_terminal_slice_noise_levels():
    t = S.table()
    ok = t["truth"] < 2000
    s15 = np.median((t["reps"][15000].std(-1) / t["truth"])[ok])
    s3 = np.median((t["reps"][3000].std(-1) / t["truth"])[ok])
    assert s15 == pytest.approx(0.0031, abs=0.001)
    assert s3 == pytest.approx(0.0065, abs=0.002)
    assert s3 > s15


def test_terminal_slice_all_methods_are_close_together():
    cur = ev.slice_curves(15000)
    opt = ev.optimum("terminal")[2]
    for m in C.METHODS:
        assert float(cur[m]["median"][15]) == pytest.approx(opt, rel=0.04), m              # bei 16 Auswertungen alle höchstens 4 % über dem Optimum
    spread = max(float(cur[m]["median"][15]) for m in C.METHODS) - min(float(cur[m]["median"][15]) for m in C.METHODS)
    assert spread < 0.04 * opt


def test_formula_ignoring_the_place_limit_and_the_rule_of_thumb():
    i, j, cost = ev.formula_choice()
    assert C.PLACES[i] == 2 and cost == pytest.approx(7070, abs=30)                         # kleinste Platzzahl, Überlast
    ri, rj, rcost = ev.rule_choice()
    assert C.PLACES[ri] == C.STAPEL_SERVERS == 5 and rcost == pytest.approx(ev.optimum("terminal")[2], abs=0.01)
    assert C.SPEEDS[rj] == pytest.approx(0.7)


def test_calibration_coverage_is_below_nominal():
    cal = ev.calibration_summary()
    assert cal[10]["coverage_median"] == pytest.approx(0.65, abs=0.1)
    assert max(v["coverage_median"] for v in cal.values()) == pytest.approx(0.86, abs=0.07)
    assert max(v["coverage_median"] for v in cal.values()) < 0.95
    assert min(v["coverage_q10"] for v in cal.values()) == pytest.approx(0.33, abs=0.15)


def test_full_network_study():
    full, formula, rule = ev.full_summary()
    assert full["random"][40]["median"] == pytest.approx(846, abs=40)
    assert full["tpe"][40]["median"] == pytest.approx(788, abs=40)
    assert full["bo"][40]["median"] == pytest.approx(846, abs=40)
    assert full["bo_log"][40]["median"] == pytest.approx(818, abs=40)
    assert rule["long_cost"] == pytest.approx(771, abs=15)
    assert formula["long_cost"] > 5000                                                        # die Formel allein wählt eine Überlast
    for m in C.METHODS:
        assert rule["long_cost"] < full[m][40]["median"], m                                  # die Faustregel schlägt jedes Suchverfahren bei 40 Auswertungen
    assert full["tpe"][30]["median"] < full["random"][30]["median"] - 30                      # TPE ab 30 Auswertungen vorn
    assert full["bo"][40]["median"] == pytest.approx(full["random"][40]["median"], abs=40)    # BO ohne log nicht besser als Zufall
    assert full["bo_log"][40]["median"] < full["bo"][40]["median"]


def test_live_run_is_reproducible_and_methods_report_true_values():
    s = ev.Settings(n=12, noise=5.0, trucks=15000, seed=35)
    a1, a2 = ev.analyse(s, "terminal"), ev.analyse(s, "terminal")
    for m in a1.runs:
        assert a1.runs[m].idx == a2.runs[m].idx
        assert a1.runs[m].truth_trace[-1] >= a1.optimum[2] - 1e-9
