"""Konstanten der Demo „Bayessche Optimierung“: das Terminal als teurer Simulator, die zwei Entscheidungen, Regler, Presets, Farben.

Zeiten in Minuten, Kosten in EUR je Stunde. Das Terminal ist das Netz aus jackson-network-demo (Gate, Kran, Stapel), hier mit einer
**Platzgrenze am Stapel** (ein fertiger Lkw blockiert den Kran, bis am Stapel ein Platz frei ist): dafür gibt es keine Formel.
"""

# --- Das Terminal -----------------------------------------------------------------------------------------------------------------------
GAMMA_PER_HOUR = 33.0            # Ankünfte je Stunde
GATE_SERVERS, KRAN_SERVERS, STAPEL_SERVERS = 4, 4, 5
GATE_MEAN, KRAN_BASE_MEAN, STAPEL_BASE_MEAN = 3.0, 2.4, 4.0      # Minuten; Kran/Stapel werden durch das Tempo geteilt
KRAN_SPEED = 0.7                 # fest in der Demo
CS2 = 2.0                        # Streuung der Dauer (Variationskoeffizient²) an allen Stationen
T_CAP = 300.0                    # Kappung der mittleren Gesamtzeit (Minuten) bei instabilem Netz

# Kosten (Annahmen der Demo, EUR je Stunde)
K_GATE, K_KRAN, K_STAPEL, K_PLACE = 20.0, 120.0, 60.0, 4.0       # je Gerät bzw. je Stellplatz; Kran/Stapel mal Tempo²
V_WAIT = 40.0                    # EUR je Lkw-Stunde im System

# --- Die zwei Entscheidungen der Scheibe (alle anderen sind fest) ---------------------------------------------------------------------------
PLACES = tuple(range(2, 14))                                    # Plätze am Stapel (wartende + in Arbeit)
SPEEDS = tuple(round(0.6 + 0.05 * i, 2) for i in range(17))     # Tempo des Stapels, 0.6 … 1.4
N_PLACES, N_SPEEDS = len(PLACES), len(SPEEDS)

# --- Die Testlandschaft (Branin) ----------------------------------------------------------------------------------------------------------
BRANIN_STEPS = 31                # Gitter 31 x 31 = 961 Einstellungen
BRANIN_NOISES = (0.0, 5.0, 20.0)  # Standardabweichung des Beobachtungsrauschens
DEFAULT_BRANIN_NOISE = 5.0

# --- Vorgerechnete Tabellen -------------------------------------------------------------------------------------------------------------
TRUCK_CHOICES = (3000, 15000)    # Lkw je Auswertung: kurzer, verrauschter oder langer Lauf
TRUCKS_LABEL = {3000: "3 000 Lkw (billig, verrauscht)", 15000: "15 000 Lkw (teurer, ruhiger)"}
TRUTH_TRUCKS, TRUTH_RUNS = 100_000, 2
REPLICATES = 30                  # unabhängige Wiederholungen je Punkt und Lauflänge in der Tabelle

# --- Regler -----------------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 6, 30, 16
N_START = 5                      # Startdesign (zufällig bzw. Sobol) bei BO
DEFAULT_TRUCKS = 15000
SEED_MAX, DEFAULT_SEED = 999999, 35
SWEEP_SEEDS = tuple(range(200000, 200020))    # 20 feste Läufe für alle Experimente
N_SWEEP = (6, 8, 10, 12, 16, 20, 24, 30)

METHODS = ("random", "tpe", "bo", "bo_log")
METHODS_BY_KIND = {"branin": ("random", "tpe", "bo"), "terminal": METHODS}      # log nur dort, wo die Werte positiv bleiben (nicht bei Gauß-Rauschen)
METHOD_LABELS = {"random": "Zufall", "tpe": "TPE (einfache Fassung)", "bo": "BO (EI)", "bo_log": "BO (EI, log-Kosten)"}
METHOD_COLORS = {"random": "#8c8c8c", "tpe": "#D68A2E", "bo": "#2F6B65", "bo_log": "#14233B"}

PRESETS = {
    "Standardfall (Voreinstellung)": {"n": DEFAULT_N, "noise": DEFAULT_BRANIN_NOISE, "trucks": 15000, "seed": DEFAULT_SEED},
    "Wenige Läufe": {"n": 8, "noise": DEFAULT_BRANIN_NOISE, "trucks": 15000, "seed": DEFAULT_SEED},
    "Rauschfrei": {"n": DEFAULT_N, "noise": 0.0, "trucks": 15000, "seed": DEFAULT_SEED},
    "Starkes Rauschen": {"n": DEFAULT_N, "noise": 20.0, "trucks": 3000, "seed": DEFAULT_SEED},
    "Großes Budget": {"n": N_MAX, "noise": DEFAULT_BRANIN_NOISE, "trucks": 15000, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "16 Auswertungen, mäßiges Rauschen (Branin: Standardabweichung 5, Terminal: Läufe à 15 000 Lkw).",
    "Wenige Läufe": "Nur 8 Auswertungen, davon 5 im Startdesign: die ersten Schritte entscheiden alles.",
    "Rauschfrei": "Branin ohne Rauschen: jede Auswertung ist exakt. Hier zeigt das Ersatzmodell, was es kann.",
    "Starkes Rauschen": "Branin mit Standardabweichung 20 und kurze Terminal-Läufe (3 000 Lkw): jede Auswertung streut stark.",
    "Großes Budget": "30 Auswertungen: bei 961 Einstellungen (Branin) bzw. 204 (Terminal) schon ein großer Teil des Gitters.",
}
NOISE_LABEL = {0.0: "0 (rauschfrei)", 5.0: "5 (mäßig)", 20.0: "20 (stark)"}

# --- Das volle Netz (sechs Entscheidungen) für die vorgerechnete Studie ------------------------------------------------------------------
FULL_GATE = tuple(range(1, 7))
FULL_KRAN = tuple(range(1, 6))
FULL_STAPEL = tuple(range(1, 9))
FULL_PLACES = PLACES
FULL_SPEED_KRAN = tuple(round(0.6 + 0.1 * i, 1) for i in range(9))
FULL_SPEED_STAPEL = FULL_SPEED_KRAN
FULL_LEVELS = (len(FULL_GATE), len(FULL_KRAN), len(FULL_STAPEL), len(FULL_PLACES), len(FULL_SPEED_KRAN), len(FULL_SPEED_STAPEL))
FULL_POOL = 4096                  # Sobol-Punkte, auf das Raster gerastert: die Kandidatenmenge der Studie
STUDY_BUDGETS = (10, 20, 30, 40)
STUDY_SEEDS = tuple(range(300000, 300020))
STUDY_TRUCKS = 15000

# Testlandschaft Hartmann6 (ohne Strukturwissen): Budgets und Kandidatenmenge
BENCH_BUDGETS = (20, 40, 60)
BENCH_POOL = 4096
BENCH_METHODS = ("random", "tpe", "bo")

COLOR_TRUTH = "#14233B"
COLOR_OBS = "#2F6B65"
COLOR_NEXT = "#D68A2E"
COLOR_BAD = "#B3463A"
