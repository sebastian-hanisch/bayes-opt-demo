"""Bayessche Optimierung: teure Simulationen mit wenigen Läufen optimieren - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Teil 1 zeigt den Mechanismus an der Testlandschaft Branin (Ersatzmodell, erwartete Verbesserung, Vergleich mit Zufall und TPE). Teil 2 prüft ihn am
Terminal-Netz als teurem Simulator: dort schlägt Wissen über das System die Suche.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import bao_constants as C
import bao_evaluation as ev
from bao_evaluation import Settings, analyse
from bao_pdf_export import build_pdf
from bao_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from bao_visualization import build_bench_chart, build_calibration_chart, build_convergence_chart, build_full_chart, build_surface_chart, build_sweep_chart

st.set_page_config(page_title="Bayessche Optimierung – Sebastian Hanisch", layout="wide")


def de(x, nd=0):
    """Dezimalkomma statt Punkt, Tausender mit Leerzeichen."""
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f} %".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings, kind):
    return analyse(settings, kind)


st.title("🧭 Bayessche Optimierung: teure Simulationen mit wenigen Läufen")
st.markdown(
    """
Eine Einstellung zu bewerten, kostet einen **Simulationslauf**, und der ist teuer; weil ein Lauf zufällig streut, ist sein Wert nie ganz sicher. Zufälliges Ausprobieren verschenkt Läufe.
**Bayessche Optimierung** lernt aus den bisherigen Läufen ein **Ersatzmodell** (einen Gauß-Prozess): eine Karte der Werte **samt Unsicherheit**. Den nächsten Lauf setzt sie dorthin,
wo am meisten zu gewinnen ist: die **erwartete Verbesserung**. Die Demo zeigt den Mechanismus erst an einer Testlandschaft und prüft ihn dann an einem **Terminal-Netz** (Gate, Kran,
Stapel mit Platzgrenze) als teurem Simulator, und dort kommt die ehrliche Überraschung: **Wissen über das System schlägt die Suche.** Die Demo ist das erste Stück der Linie
**Bayessche Optimierung**; im Unterschied zu den Fall-Demos zeigt sie ein einziges Verfahren an einem wachsenden Beispiel.

Weiter unten: fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Bayessche Optimierung", expanded=True):
    st.markdown(
        """
Nach einem kleinen **Startdesign** (hier 5 zufällige Läufe) wiederholt sich je Schritt:

1. **Lernen:** Der Gauß-Prozess passt Längenmaße und Rauschen an die bisherigen Läufe an. Ergebnis: für **jede** Einstellung ein Mittelwert und eine Unsicherheit.
2. **Bewerten:** Für jede noch nicht gelaufene Einstellung die **erwartete Verbesserung** gegenüber dem bisher Besten: groß, wo der Mittelwert niedrig **oder** die Unsicherheit hoch ist.
3. **Ausführen:** Der Lauf mit der größten erwarteten Verbesserung wird gerechnet, sein Ergebnis kommt zu den Daten.
4. **Empfehlen:** Nach dem Budget gilt die beste Beobachtung als Empfehlung. Bewertet wird sie an der **Wahrheit**, nicht am verrauschten Messwert.

**Verglichen** wird mit Zufall (gleiches Budget) und **TPE** (einer einfachen Fassung der in der Praxis verbreiteten Parzen-Dichte-Suche). Am Terminal kommen zwei Wege **ohne einen
einzigen Lauf** dazu: die **Formel** (sie rechnet das Netz ohne Platzgrenze und kennt den Wert der Plätze nicht) und die **Formel mit Faustregel** eines Kenners: so viele Plätze wie
Stapelgeräte, den Rest rechnet die Formel.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Budget**")
    n = st.slider("Auswertungen (Simulationsläufe)", *bounds("n_slider"), key="n_slider", step=1,
                  help="Wie viele Läufe die Suche bezahlen darf. Gilt für beide Landschaften.")
    st.markdown("**Teil 1: Testlandschaft Branin**")
    noise = st.select_slider("Rauschen der Auswertungen", C.BRANIN_NOISES, key="noise_select", format_func=lambda z: C.NOISE_LABEL[z],
                             help="Standardabweichung des Gauß-Rauschens auf jeder Auswertung. Die Werte der Landschaft liegen zwischen 0,4 und rund 300.")
    st.markdown("**Teil 2: Terminal-Netz**")
    trucks = st.select_slider("Länge eines Laufs", C.TRUCK_CHOICES, key="trucks_select", format_func=lambda t: C.TRUCKS_LABEL[t],
                              help="Kurze Läufe sind billig, aber streuen stark; lange sind ruhiger. Die Tabelle enthält zu beiden je 30 echte Wiederholungen je Einstellung.")
    seed = st.number_input("Zufalls-Seed des Laufs", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "noise_select": float(noise), "trucks_select": int(trucks), "seed_input": int(seed)})

settings = Settings(n=int(n), noise=float(noise), trucks=int(trucks), seed=int(seed))


def method_metrics(a, with_formula):
    fc = ev.final_costs(a)
    opt = a.optimum[2]
    cols = st.columns(len(a.runs) + (2 if with_formula else 0))
    for col, k in zip(cols, a.runs):
        if a.kind == "branin":
            col.metric(C.METHOD_LABELS[k], f"{de(fc[k] - opt, 2)}", delta=None,
                       help="Regret: wahrer Wert der empfohlenen Einstellung minus Minimum des Gitters. 0 wäre die beste Einstellung.")
        else:
            col.metric(C.METHOD_LABELS[k], f"{de(fc[k])} EUR/h", delta=f"{ev.regret_pct(fc[k], opt):+.1f} % gegen Optimum".replace(".", ","), delta_color="inverse",
                       help="Wahre Kosten (lange Simulation) der Einstellung, die dieses Verfahren nach dem Budget empfiehlt.")
    if with_formula:
        cols[-2].metric("Formel", f"{de(a.formula[2])} EUR/h", delta=f"{ev.regret_pct(a.formula[2], opt):+.1f} % gegen Optimum".replace(".", ","),
                        delta_color="inverse", help="Die Formel kennt die Platzgrenze nicht (Plätze kosten nur) und wählt ohne Lauf. Wahre Kosten ihrer Wahl.")
        cols[-1].metric("Formel + Faustregel", f"{de(a.rule[2])} EUR/h", delta=f"{ev.regret_pct(a.rule[2], opt):+.1f} % gegen Optimum".replace(".", ","),
                        delta_color="inverse", help="Plätze = Zahl der Stapelgeräte, Tempo nach der Formel; ohne einen Lauf. Wahre Kosten.")
    return fc


def surface_block(a, key):
    st.markdown("**Was das Ersatzmodell glaubt** – Schritt für Schritt nachvollziehen")
    c1, c2 = st.columns(2)
    with c1:
        step = st.slider("Nach wie vielen Auswertungen?", C.N_START, C.N_MAX, C.N_START + 3, key=f"surface_step_{key}",
                         help="Bis zur gewählten Zahl der Auswertungen; bei mehr als dem Budget gilt das Budget.")
    with c2:
        methods = [k for k in ("bo", "bo_log") if k in a.runs]
        method = st.radio("Verfahren", methods, format_func=lambda k: C.METHOD_LABELS[k], key=f"surface_method_{key}", horizontal=True)
    step = int(min(step, a.settings.n))
    st.plotly_chart(build_surface_chart(a, step, method), width="stretch", key=f"surface_chart_{key}_{method}_{step}")
    st.caption(f"Nach {step} Auswertungen. Links die Wahrheit, in der Mitte der Mittelwert des Ersatzmodells aus den bisher ausgewerteten Einstellungen (grüne Punkte, Nummer = "
               "Reihenfolge), rechts, wo es als Nächstes hinsähe. Werte über der Farbskala sind gekappt.")


# --- Teil 1: Branin ------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Teil 1: Der Mechanismus an einer Testlandschaft")
b = _analysis(settings, "branin")
b_opt = b.optimum[2]
st.caption(
    "Branin: zwei Eingaben (x1 von −5 bis 10, x2 von 0 bis 15), drei gleich tiefe Mulden, Minimum 0,398. Das Gitter hat 31 × 31 = 961 Einstellungen; jede Auswertung bekommt Gauß-Rauschen "
    f"mit Standardabweichung {str(settings.noise).replace('.', ',')}."
)
bfc = method_metrics(b, with_formula=False)

st.markdown("**Wie sich die Empfehlung mit jedem Lauf verbessert**")
st.plotly_chart(build_convergence_chart(b), width="stretch", key="convergence_branin")
surface_block(b, "branin")

st.markdown("### ❓ Kernfrage 1: Lohnt das Ersatzmodell gegenüber dem Zufall?")
b_bo = bfc["bo"]
if (b_bo - b_opt) < 0.5 * (bfc["random"] - b_opt):
    st.success(
        f"**Ja, in diesem Lauf:** nach {settings.n} Auswertungen Regret {de(b_bo - b_opt, 2)} (BO) gegen {de(bfc['random'] - b_opt, 2)} (Zufall) und {de(bfc['tpe'] - b_opt, 2)} (TPE)."
    )
elif (b_bo - b_opt) > 1.5 * (bfc["random"] - b_opt) + 0.05:
    st.info(f"**Hier nicht:** Regret {de(b_bo - b_opt, 2)} (BO) gegen {de(bfc['random'] - b_opt, 2)} (Zufall). Bei wenigen Auswertungen und starkem Rauschen entscheidet oft das Startdesign.")
else:
    st.info(f"**Gleichauf:** Regret {de(b_bo - b_opt, 2)} (BO) gegen {de(bfc['random'] - b_opt, 2)} (Zufall).")
st.caption("Ein einzelner Lauf zeigt nur, was bei diesem Seed passiert ist; Experiment 1 mittelt über 20 feste Läufe.")

st.subheader("🔬 Experiment 1: Wie viele Auswertungen braucht es?")
bc = ev.branin_curves(settings.noise)
st.caption(f"Median und mittlere Hälfte der 20 festen Läufe; Rauschen wie in der Seitenleiste ({str(settings.noise).replace('.', ',')}), Regret logarithmisch.")
st.plotly_chart(build_sweep_chart(bc, C.N_MAX, b_opt, "branin"), width="stretch", key="sweep_branin")
nn = settings.n - 1
bm = {k: float(bc[k]["median"][nn]) - b_opt for k in bc}
b30 = {k: float(bc[k]["median"][C.N_MAX - 1]) - b_opt for k in bc}
st.info(
    f"Bei {settings.n} Auswertungen liegt der Median des Regrets bei {de(bm['random'], 2)} (Zufall), {de(bm['tpe'], 2)} (TPE) und {de(bm['bo'], 2)} (BO); bei {C.N_MAX} Auswertungen "
    f"bei {de(b30['random'], 2)}, {de(b30['tpe'], 2)} und {de(b30['bo'], 2)}. "
    + ("**BO liegt hier schon bei diesem Budget klar vorn.**" if bm["bo"] < 0.7 * bm["random"] else
       "**Bei diesem Budget ist der Vorsprung noch klein: das Ersatzmodell braucht erst einige Läufe, bevor es sich lohnt.**")
    + (" Mit mehr Läufen wächst er." if b30["bo"] < 0.7 * b30["random"] else "")
)

st.subheader("🔬 Experiment 2: Sechs Eingaben statt zwei")
bench, bopt = ev.bench_summary()
st.caption(
    f"Testfunktion Hartmann6 (sechs Eingaben, vier Mulden), rauschfrei, Kandidatenmenge {C.BENCH_POOL} Punkte, 20 feste Läufe. Regret = bester gefundener Wert minus bester Wert der "
    "Kandidatenmenge."
)
st.plotly_chart(build_bench_chart(bench), width="stretch", key="bench_chart")
st.info(
    f"Bei 40 Auswertungen: Zufall {de(bench['random'][40]['median'], 2)}, TPE {de(bench['tpe'][40]['median'], 2)}, BO {de(bench['bo'][40]['median'], 2)} (Median des Regrets; 0 wäre die beste "
    "Stelle der Kandidatenmenge). Mit mehr Eingaben wächst der Raum, den der Zufall absuchen müsste; das Ersatzmodell nutzt die Glätte der Landschaft."
)

st.markdown("---")

# --- Teil 2: Terminal ----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🏭 Teil 2: Der Realitätscheck am Terminal")
st.markdown(
    "Jetzt der **teure Simulator**: ein Terminal mit 4 Gate-Spuren, 4 Kränen und 5 Stapelgeräten (33 Lkw je Stunde), in dem ein fertiger Lkw den Kran blockiert, bis am Stapel ein Platz frei "
    "ist. Zwei Entscheidungen: **Plätze am Stapel** (2 bis 13) und **Tempo des Stapels** (0,6 bis 1,4). Zielgröße: **Gesamtkosten je Stunde** aus Geräten, Stellplätzen und Wartezeit."
)
t = _analysis(settings, "terminal")
t_opt = t.optimum[2]
t_fc = method_metrics(t, with_formula=True)
st.caption(
    f"Wahres Optimum der Scheibe: **{C.PLACES[t.optimum[0]]} Plätze, Tempo {str(C.SPEEDS[t.optimum[1]]).replace('.', ',')}**, {de(t_opt)} EUR/h. Die Auswertungen stammen aus einer "
    f"Tabelle echter Simulationen (30 Wiederholungen je Einstellung, {de(settings.trucks)} Lkw je Lauf)."
)
st.markdown("**Wie sich die Empfehlung mit jedem Lauf verbessert**")
st.plotly_chart(build_convergence_chart(t), width="stretch", key="convergence_terminal")
surface_block(t, "terminal")
st.download_button("📄 Auswertung als PDF", data=build_pdf(t), file_name="bayes_opt_auswertung.pdf", mime="application/pdf", key="pdf_button")

st.markdown("### ❓ Kernfrage 2: Braucht man die Suche überhaupt?")
t_bo = min(t_fc["bo"], t_fc["bo_log"])
if t.rule[2] <= t_bo * 1.005:
    st.warning(
        f"**Nein, nicht hier:** Die **Formel mit Faustregel** liegt bei {de(t.rule[2])} EUR/h, BO nach {settings.n} Läufen bei {de(t_bo)} EUR/h, der Zufall bei "
        f"{de(t_fc['random'])} EUR/h. Wer die Struktur des Systems kennt, braucht die Suche nicht: **Bayessche Optimierung ersetzt kein Wissen.**"
    )
else:
    st.info(f"Die Formel mit Faustregel liegt bei {de(t.rule[2])} EUR/h, BO bei {de(t_bo)} EUR/h.")
if t.formula[2] > 1.15 * t_opt:
    st.warning(
        f"Die **Formel allein** ohne Platzgrenze kostet {de(t.formula[2])} EUR/h, also {ev.regret_pct(t.formula[2], t_opt):.0f} % über dem Optimum: "
        "sie wählt die kleinste Platzzahl, weil ihr Platzmangel nichts kostet, und das Terminal blockiert. Erst die Faustregel macht sie brauchbar."
    )
st.caption("Ein einzelner Lauf zeigt nur, was bei diesem Seed passiert ist; die Experimente unten mitteln über 20 feste Läufe.")

st.subheader("🔬 Experiment 3: Wie viele Auswertungen braucht es am Terminal?")
curves = ev.slice_curves(settings.trucks)
st.caption(f"Median und mittlere Hälfte der 20 festen Läufe der Scheibe; Länge eines Laufs wie in der Seitenleiste ({de(settings.trucks)} Lkw).")
st.plotly_chart(build_sweep_chart(curves, C.N_MAX, t_opt, "terminal", t.rule[2]), width="stretch", key="sweep_terminal")
med = {k: float(curves[k]["median"][settings.n - 1]) for k in C.METHODS}
st.info(
    f"Bei {settings.n} Auswertungen liegt der Median der Empfehlung bei {de(med['random'])} EUR/h (Zufall), {de(med['tpe'])} (TPE), {de(med['bo'])} (BO) und {de(med['bo_log'])} "
    f"(BO mit log-Kosten); das Optimum liegt bei {de(t_opt)}. **Alle Verfahren liegen dicht beieinander**: Das Tal der Kosten ist breit und flach, schon zufällige Einstellungen "
    "liegen nahe am Optimum. Nur die Überlast (Wahrheit oben, über 7 000 EUR/h) ist wirklich schlecht."
)

st.subheader("🔬 Experiment 4: Kann man dem Ersatzmodell glauben?")
cal = ev.calibration_summary()
st.caption("Je Beobachtungszahl 20 feste Läufe mit zufälligen Einstellungen (Lauflänge 15 000 Lkw): Anteil der 204 Gitterpunkte, deren wahre Kosten im nominalen 95-%-Intervall des Modells liegen.")
st.plotly_chart(build_calibration_chart(cal), width="stretch", key="calibration_chart")
lo_n = min(cal, key=lambda k: cal[k]["coverage_median"])
st.info(
    f"Die Abdeckung liegt im Median zwischen {pct(min(v['coverage_median'] for v in cal.values()))} (bei {lo_n} Beobachtungen) und "
    f"{pct(max(v['coverage_median'] for v in cal.values()))}; im schlechtesten Zehntel der Läufe bis herab auf {pct(min(v['coverage_q10'] for v in cal.values()))}. "
    "**Das Intervall des Ersatzmodells ist keine Garantie**: die erwartete Verbesserung rechnet mit einer Unsicherheit, die nur so gut ist wie das Modell."
)

st.subheader("🔬 Experiment 5: Und im ganzen Netz mit sechs Entscheidungen?")
full, formula, rule = ev.full_summary()
st.caption(
    f"Nicht aus der Tabelle, sondern echte Simulationen à 15 000 Lkw: Geräte je Station, Plätze und beide Tempi (rund 230 000 Einstellungen, Kandidatenmenge aus "
    f"{ev.study()['meta']['pool']} Punkten). 20 feste Läufe je Verfahren; die Empfehlung wird in einer langen Simulation nachbewertet."
)
st.plotly_chart(build_full_chart(full, rule), width="stretch", key="full_chart")
b_hi = max(C.STUDY_BUDGETS)
rows = "\n".join(
    f"| {C.METHOD_LABELS[k]} | " + " | ".join(f"{de(full[k][b]['median'])}" for b in C.STUDY_BUDGETS) + f" | {de(full[k][b_hi]['q90'])} |" for k in C.METHODS
)
st.markdown(
    "| Verfahren | " + " | ".join(f"{b} Aus. (Median)" for b in C.STUDY_BUDGETS) + f" | {b_hi} Aus. (90-%-Quantil) |\n|---|" + "---|" * (len(C.STUDY_BUDGETS) + 1) + "\n" + rows
)
st.info(
    f"Bei {b_hi} Auswertungen: Zufall {de(full['random'][b_hi]['median'])}, TPE {de(full['tpe'][b_hi]['median'])}, BO {de(full['bo'][b_hi]['median'])}, BO mit log-Kosten "
    f"{de(full['bo_log'][b_hi]['median'])} EUR/h. Die Formel ohne Platzgrenze wählt {de(formula['long_cost'])} EUR/h, die **Formel mit Faustregel** {de(rule['long_cost'])} EUR/h, ohne einen Lauf. "
    "Das ist weniger als jedes Suchverfahren in 40 Läufen erreicht."
)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Man weiß nichts über die Struktur** | Dann ist die Suche ihr Geld wert (Branin, Hartmann6). Weiß man etwas (Formel, Faustregel, Erfahrung), schlägt es oft jede Suche mit wenigen Läufen, wie das Terminal zeigt. | Formel als Vorwissen im Modell |
| **Die Zielfunktion ist glatt** | Klippen und Plateaus (hier: Überlast) verzerren die Skala; ein Gauß-Prozess mit gleichem Längenmaß überall passt schlecht. Der Logarithmus hilft nur teilweise. | Transformationen, Modelle mit wechselnder Glätte |
| **Das Rauschen ist gleichmäßig** | Kurze Läufe streuen bei Last viel stärker als bei Leerlauf. Das Modell lernt **ein** Rauschen für alle Einstellungen. | Verrauschte Auswertungen |
| **Wenige Entscheidungen** | Das Terminal-Gitter hat zwei, die Studie sechs, die Testfunktion Hartmann6 sechs. Wie weit das mit der Zahl der Entscheidungen trägt, ist hier nicht gemessen. | Hohe Dimensionen |
| **Das Ersatzmodell ist gut abgestimmt** | Längenmaße werden aus wenigen Läufen gelernt; die Unsicherheit ist nicht kalibriert (Experiment 4). | Die Wahl des Ersatzmodells |
| **Läufe nacheinander** | Jede Auswertung wartet auf die vorige. Parallele Läufe brauchen andere Auswahlregeln. | Parallele Auswertungen |
| **Es gibt keine Nebenbedingung** | Eine Grenze auf der Wartezeit würde die Suche an den Rand führen. | Nebenbedingungen |
    """
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Ersatzmodell.** Gauß-Prozess mit Matérn-5/2-Kern, je Eingabe ein Längenmaß $\ell_d$ und Signalvarianz $\sigma_f^2$:

$$k(x, x') = \sigma_f^2\Bigl(1 + \sqrt5\, r + \tfrac53 r^2\Bigr)\exp\bigl(-\sqrt5\, r\bigr), \qquad r = \sqrt{\textstyle\sum_d (x_d - x'_d)^2/\ell_d^2}$$

**Vorhersage.** Mit Beobachtungen $y = f(X) + \varepsilon$, $\varepsilon \sim \mathcal N(0, \sigma_n^2)$ und $K = k(X, X) + \sigma_n^2 I$:

$$\mu(x) = k(x, X)\,K^{-1} y$$

$$s^2(x) = k(x, x) - k(x, X)\,K^{-1} k(X, x)$$

**Lernen.** Längenmaße, $\sigma_f^2$ und $\sigma_n^2$ maximieren die Randwahrscheinlichkeit (Ausgabe vorher standardisiert):

$$\log p(y \mid X) = -\tfrac12 y^\top K^{-1} y - \tfrac12 \log|K| - \tfrac n2 \log 2\pi$$

**Erwartete Verbesserung** (Minimierung, bester Wert $f^\*$ = kleinster Mittelwert an den Beobachtungen), $z = (f^\* - \mu)/s$:

$$\mathrm{EI}(x) = (f^\* - \mu)\,\Phi(z) + s\,\varphi(z)$$

**Auswahl.** $x_{n+1} = \arg\max_x \mathrm{EI}(x)$ über die noch nicht gelaufenen Einstellungen.

**TPE (Vergleich).** Die besten 25 % der Beobachtungen bilden die Dichte $l$, die übrigen $g$ (Gauß-Kerne); gewählt wird $\arg\max_x l(x)/g(x)$.

**Literatur.** D. R. Jones, M. Schonlau, W. J. Welch (1998), *Efficient Global Optimization of Expensive Black-Box Functions*, Journal of Global Optimization 13, 455–492 (EI für
Gauß-Prozesse); C. E. Rasmussen, C. K. I. Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press (Kern, Randwahrscheinlichkeit); J. Bergstra, R. Bardenet, Y. Bengio,
B. Kégl (2011), *Algorithms for Hyper-Parameter Optimization*, NeurIPS (TPE).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
