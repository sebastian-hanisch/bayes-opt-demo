# Bayessche Optimierung: teure Simulationen mit wenigen Läufen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-bayes-opt-demo.streamlit.app/)**

Erstes Stück der **Bayessche-Optimierung-Linie** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es zeigt das Verfahren (Gauß-Prozess als Ersatzmodell, erwartete Verbesserung als Auswahlregel) erst an einer Testlandschaft und prüft es dann am **Terminal-Netz** aus der Linie
Warteschlangentheorie (Gate, Kran, Stapel) als teurem Simulator, hier mit einer Platzgrenze am Stapel. Im Unterschied zu den Fall-Demos zeigt die Demo ein einziges Verfahren an einem
wachsenden Beispiel.

## Warum dieses Problem

Wo ein Simulationslauf Minuten kostet und streut, ist Ausprobieren teuer. Die Kernfrage: **Wann lohnt ein lernendes Ersatzmodell gegenüber Zufall, gegenüber einer einfacheren
Suche (TPE) und gegenüber Wissen über das System?** Die Antwort ist zweigeteilt, und die zweite Hälfte ist unbequem.

## Modell

- **Ersatzmodell:** Gauß-Prozess mit Matérn-5/2-Kern, je Eingabe ein Längenmaß, Rauschvarianz mitgelernt (Randwahrscheinlichkeit, L-BFGS-B mit drei festen Startpunkten); Ausgabe standardisiert.
- **Auswahl:** erwartete Verbesserung (EI) über die noch nicht gelaufenen Einstellungen der Kandidatenmenge; Startdesign 5 zufällige Läufe (in der Studie 10). Empfehlung = beste Beobachtung.
- **Vergleich:** Zufall (ohne Zurücklegen), TPE (eigene einfache Fassung: besten 25 % gegen die übrigen, Gauß-Kerne), BO mit log-Werten (nur dort, wo die Werte positiv sind).
- **Landschaften:** (1) Branin auf einem 31 × 31-Gitter mit Gauß-Rauschen (0, 5, 20); (2) Hartmann6, sechs Eingaben, rauschfrei; (3) das Terminal als **Scheibe** (Plätze am Stapel
  × Tempo des Stapels, 12 × 17 = 204 Einstellungen, Auswertungen aus einer Tabelle echter Simulationen) und als **volles Netz** mit sechs Entscheidungen (Geräte je Station, Plätze, beide Tempi).
- **Terminal:** 33 Lkw je Stunde, Gate 4 Spuren (3 min), Kran 4 Geräte (2,4 min durch Tempo 0,7), Stapel 5 Geräte (4 min durch Tempo); Streuung der Dauer cs² = 2;
  Platzgrenze am Stapel mit Blockieren nach Abfertigung. Kosten je Stunde: 20 je Gate-Spur, 120 je Kran, 60 je Stapelgerät (die beiden letzten mal Tempo²), 4 je Stellplatz, 40 je
  Lkw-Stunde im System. **Annahmen der Demo**, keine Marktpreise.
- Formeln im 📐-Expander, Kern in `bao_algorithm.py`, Simulator in `bao_network.py` (Kopie aus `jackson-network-demo`).

## Methodik

1. **Teil 1, Branin live:** ein Lauf je Verfahren mit einstellbarem Budget, Rauschen und Seed; das Ersatzmodell lässt sich Schritt für Schritt ansehen.
2. **Experiment 1, Experiment 2:** je 20 feste Läufe, Branin über das Budget und Hartmann6 bei 20/40/60 Auswertungen (vorgerechnet, `generate_study.py`).
3. **Teil 2, Terminal live:** ein Lauf je Verfahren; die Auswertungen sind **echte Simulationen** aus einer Tabelle (30 Wiederholungen je Einstellung und Lauflänge, `generate_precomputed.py`),
   die Wahrheit je Einstellung ist der Mittelwert zweier Läufe à 100 000 Lkw. Dazu zwei Wege ohne Lauf: die **Formel** (Zerlegung nach Whitt, kennt die Platzgrenze nicht) und die
   **Formel mit Faustregel** (Plätze = Zahl der Stapelgeräte, Tempo nach der Formel).
4. **Experimente 3 bis 5:** Terminal-Scheibe über das Budget, Kalibrierung des Gauß-Prozesses, volles Netz mit sechs Entscheidungen (Kandidatenmenge 4 095 gerasterte Sobol-Punkte,
   echte Simulationen à 15 000 Lkw, Nachbewertung der Empfehlung mit 2 × 100 000 Lkw).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Medianwerte über je 20 feste Läufe; „Regret“ = wahrer Wert der Empfehlung minus bester Wert des Gitters bzw. der Kandidatenmenge.

| Frage | Befund |
|---|---|
| Branin ohne Rauschen: wer erreicht das Gitter-Optimum? | Bei 30 Auswertungen **BO und TPE: Regret 0** (das Optimum), **Zufall 0,71**. |
| Branin, Rauschen 5: ab wann lohnt BO? | **Bei 16 Auswertungen kaum**: Regret 3,66 (Zufall), 2,91 (TPE), 3,13 (BO). **Bei 30**: 3,09 (Zufall), 1,56 (TPE), **0,94 (BO)**. Das Ersatzmodell braucht erst einige Läufe. |
| Branin, Rauschen 20 | Bei 30 Auswertungen Zufall **13,5**, TPE **5,1**, BO **5,2**: die Suchverfahren senken den Regret auf rund 40 %, TPE ist gleichauf mit BO. |
| Sechs Eingaben (Hartmann6, rauschfrei) | Bei 40 Auswertungen Regret Zufall **1,45**, TPE **0,56**, BO **0,08**; bei 60 Zufall 1,09, TPE 0,15. **Hier ist der Abstand groß.** |
| Terminal-Scheibe (204 Einstellungen) | Optimum **5 Plätze, Tempo 0,7: 791 EUR/h**; **ein Fünftel aller Einstellungen liegt höchstens 5 % darüber**, 8 % sind Überlast. Bei 16 Auswertungen liegen alle vier Verfahren im Median höchstens **4 %** über dem Optimum: **kein Unterschied**. |
| Die Formel allein | Sie kennt den Wert der Plätze nicht und wählt die kleinste Platzzahl (2): **7 070 EUR/h**, rund **9-mal** so teuer wie das Optimum (Überlast). |
| Formel + Faustregel (Plätze = Stapelgeräte) | In der Terminal-Scheibe **das Optimum selbst, ohne einen Lauf**: 791 EUR/h. Im vollen Netz **771 EUR/h** (Nachbewertung), besser als jedes Suchverfahren bei 40 Auswertungen. |
| Volles Netz, 40 Auswertungen | Median der wahren Kosten: Zufall **846**, TPE **788**, BO **846**, BO mit log-Kosten **818** EUR/h. BO ohne log ist hier **nicht besser als Zufall**; TPE liegt ab 30 Auswertungen vorn. |
| Kann man dem Ersatzmodell glauben? | Das nominale 95-%-Intervall deckt die wahren Kosten im Median zu **65 %** (10 Beobachtungen) bis **86 %** (30) ab; im schlechtesten Zehntel der Läufe bis **33 %**. |
| Rauschen der Läufe | Streuung einer Auswertung im Median **0,3 %** der Kosten (15 000 Lkw) bzw. **0,65 %** (3 000 Lkw) auf stabilen Einstellungen. |

## Ehrliche Grenzen

- **Das Terminal ist ein Gegenbeispiel, kein Beleg für BO.** In der Scheibe liegen bei 16 Auswertungen alle vier Verfahren im Median höchstens 4 % über dem Optimum; richtig teuer ist vor allem die Überlast. Wissen über das Netz (Formel plus eine Faustregel) schlägt im vollen Netz jede Suche mit 40 Läufen, gegen TPE knapp (771 gegen 788 EUR/h). BO liegt auf Branin und Hartmann6 vor dem Zufall (bei Rauschen erst mit mehr Budget); im vollen Netz ist BO ohne log nicht besser als Zufall.
- **Die Faustregel ist die Entscheidung eines Kenners.** Sie wurde **nach** dem Blick auf das Ergebnis der Formel als naheliegende Ergänzung gewählt; ein Vergleich „Formel gegen Suche“ ohne sie
  hätte die Suche besser aussehen lassen, als sie ist. Das zeigt die Demo daher beide Formeln.
- **TPE und Successive Halving sind einfache eigene Fassungen**, nicht Optuna oder Hyperband; ein Urteil über „TPE“ im Allgemeinen folgt daraus nicht.
- **Eigene BO-Implementierung** (Kandidatenmenge statt stetiger Optimierung der Akquisition, drei Startpunkte der Hyperparameter-Suche): Zahlen anderer Bibliotheken können abweichen.
- **Die Auswertungen der Scheibe sind eine Tabelle**, aus der je Einstellung eine der 30 Wiederholungen ohne Zurücklegen gezogen wird; zieht man mehr als 30 mal dieselbe Einstellung, beginnt der
  Vorrat von vorn.
- **Kleine Stichprobe:** 20 feste Läufe je Zelle; Unterschiede unter etwa 10 % des Regrets sind Zufall. Einzelne Läufe zeigen nur, was bei diesem Seed geschah.
- **Zwei Entscheidungen in der Scheibe, sechs im Netz.** Wie das mit mehr Entscheidungen, verrauschten Nebenbedingungen oder parallelen Läufen aussieht, ist nicht Teil dieser Demo.

## Verifikation

- **Kern:** Kern, Erwartete Verbesserung, Suchverfahren an Mini-Instanzen von Hand (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_bayes_opt.py`, anderer Rechenweg):
  - Gauß-Prozess-Vorhersage gegen **scikit-learn** bei den gelernten Hyperparametern; Randwahrscheinlichkeit nach dem Lernen mindestens so hoch wie bei festen Standardwerten.
  - Erwartete Verbesserung gegen **numerische Integration** (scipy) und Monte-Carlo.
  - Simulator: **M/M/1** gegen die geschlossene Formel, exponentielles Tandem gegen die **Jackson-Formel**, Platzgrenze von Hand (Verweilzeiten 4 und 5 Minuten, Mittel 4,5).
  - Suche gegen **Aufzählung** kleiner Gitter.
- **Tabelle und Szenario:** Handrechnung der Kosten, Ziehen ohne Zurücklegen, Neurechnung einer Zelle gegen die Wahrheit (`tests/test_scenario.py`).
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `bao_algorithm.py` | Kern: Gauß-Prozess, erwartete Verbesserung, Zufall, TPE, BO |
| `bao_scenario.py` | Terminal-Kosten, Simulation, Tabelle, Branin-Feld, volles Netz |
| `bao_network.py`, `bao_formulas.py` | Kopie von Simulator und Formeln aus `jackson-network-demo` |
| `bao_benchmarks.py` | Branin und Hartmann6 |
| `bao_evaluation.py` | Läufe, Kennzahlen, vorgerechnete Studien |
| `bao_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `bao_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `bao_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `bao_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `generate_precomputed.py`, `precomputed_slice.json` | Tabelle der Scheibe (Wahrheit und Wiederholungen) |
| `generate_study.py`, `precomputed_study.json` | Studien (Branin, Kalibrierung, volles Netz, Hartmann6) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest und scikit-learn |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, Szenario, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Stetige Optimierung der Akquisitionsfunktion** (Mehrfachstart-Gradientenabstieg): die Kandidatenmenge ist endlich und aufzählbar, damit sich jedes Ergebnis exakt prüfen lässt.
- **Andere Akquisitionsfunktionen** (UCB, Thompson, Wissensgradient, Entropie-Suche) und **andere Kerne**: eigene Themen, hier nur EI mit Matérn-5/2.
- **Echte Hyperband- oder Optuna-Läufe, BoTorch.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
