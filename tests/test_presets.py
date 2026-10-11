"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Einrasten."""

import bao_constants as C
import bao_evaluation as ev
import bao_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_bounds():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and preset["trucks"] in C.TRUCK_CHOICES and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_value_snaps_to_an_allowed_run_length():
    assert P.snap_to_choice(14000, C.TRUCK_CHOICES) == 15000
    assert P.snap_to_choice(5000, C.TRUCK_CHOICES) == 3000
    assert P.snap_to_choice(3000, C.TRUCK_CHOICES) == 3000


def test_start_design_fits_into_every_budget():
    assert C.N_START <= C.N_MIN
