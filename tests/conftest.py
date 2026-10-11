"""Gemeinsame Fixtures: Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def toy():
    """Eindimensionale Aufgabe: 21 Kandidaten auf [0, 1], f(x) = (x - 0,3)^2 + 1, Optimum bei Index 6 (x = 0,3), rauschfrei."""
    cand = np.linspace(0.0, 1.0, 21)[:, None]
    f = (cand[:, 0] - 0.3) ** 2 + 1.0
    return cand, f
