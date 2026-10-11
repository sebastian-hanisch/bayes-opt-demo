"""Glatte Testlandschaft ohne Strukturwissen: Hartmann6 (Dixon und Szegö 1978), sechs Eingaben auf [0, 1], rauschfrei, Minimierung."""

import numpy as np

_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650],
                      [4047, 8828, 8732, 5743, 1091, 381]])
_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
GLOBAL_MIN = -3.32237      # bekanntes Minimum (Literatur)


def hartmann6(x):
    """Hartmann-Funktion in sechs Dimensionen: vier Mulden, eine tiefste. x auf [0, 1]^6."""
    x = np.asarray(x, float)
    return -float((_ALPHA * np.exp(-(_A * (x - _P) ** 2).sum(1))).sum())


# --- Branin: zwei Eingaben, drei gleich tiefe Mulden (die klassische Testlandschaft der globalen Optimierung) -------------------------------

BRANIN_X1 = (-5.0, 10.0)
BRANIN_X2 = (0.0, 15.0)
BRANIN_MIN = 0.397887      # bekanntes Minimum (Literatur), an drei Stellen


def branin(x1, x2):
    """Branin-Hoo-Funktion auf x1 in [-5, 10], x2 in [0, 15] (Minimierung)."""
    b, c, r, s, t = 5.1 / (4 * np.pi ** 2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8 * np.pi)
    return (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def branin_grid(steps):
    """Gitter von steps x steps Punkten: (x1-Werte, x2-Werte, Wahrheit [x2-Index, x1-Index])."""
    x1 = np.linspace(BRANIN_X1[0], BRANIN_X1[1], steps)
    x2 = np.linspace(BRANIN_X2[0], BRANIN_X2[1], steps)
    X1, X2 = np.meshgrid(x1, x2)
    return x1, x2, branin(X1, X2)
