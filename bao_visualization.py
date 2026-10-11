"""Plotly-Abbildungen (alle Achsen fest, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import bao_algorithm as A
import bao_constants as C
import bao_scenario as S

SURFACE = {   # je Landschaft: Farbbereich (Werte darüber zählen als "sehr hoch"), Achsentitel, Einheit
    "branin": {"zmin": 0.0, "zmax": 100.0, "x": "x1", "y": "x2", "unit": "Funktionswert"},
    "terminal": {"zmin": 750.0, "zmax": 1500.0, "x": "Tempo Stapel", "y": "Plätze am Stapel", "unit": "EUR/h"},
}


def _fixed(fig, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2), **layout)
    return fig


def _heat(a, z, colorscale, zmin=None, zmax=None):
    return go.Heatmap(z=z, x=a.x_values, y=a.y_values, colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=False,
                      hovertemplate="%{x}<br>%{y}<br>%{z:.1f}<extra></extra>")


def build_surface_chart(a, step, method="bo"):
    """Drei Flächen: Wahrheit, was das Ersatzmodell nach `step` Auswertungen glaubt (Mittelwert), erwartete Verbesserung."""
    cfg = SURFACE[a.kind]
    run = a.runs[method]
    shape = a.truth.shape
    pts = S.grid_points(shape)
    cand = S.to_unit(pts, shape)
    idx = run.idx[:step]
    mu, sd, ei, _ = A.gp_state(cand, idx, run.y[:step], log=(method == "bo_log"))
    clip = lambda z: np.clip(z, cfg["zmin"], cfg["zmax"])
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Wahrheit", "Ersatzmodell (Mittel)", "Erwartete Verbesserung"), horizontal_spacing=0.06)
    fig.add_trace(_heat(a, clip(a.truth), "YlGnBu_r", cfg["zmin"], cfg["zmax"]), row=1, col=1)
    fig.add_trace(_heat(a, clip(mu.reshape(shape)), "YlGnBu_r", cfg["zmin"], cfg["zmax"]), row=1, col=2)
    fig.add_trace(_heat(a, ei.reshape(shape), "Oranges"), row=1, col=3)
    px = [a.x_values[int(pts[i][1])] for i in idx]
    py = [a.y_values[int(pts[i][0])] for i in idx]
    for col in (1, 2, 3):
        fig.add_trace(go.Scatter(x=px, y=py, mode="markers+text", text=[str(k + 1) for k in range(len(idx))], textposition="top center", textfont=dict(size=9),
                                 marker=dict(size=8, color=C.COLOR_OBS, line=dict(color="white", width=1)), name="ausgewertet", showlegend=(col == 1),
                                 hovertemplate="Auswertung %{text}<extra></extra>"), row=1, col=col)
    oi, oj, _ = a.optimum
    fig.add_trace(go.Scatter(x=[a.x_values[oj]], y=[a.y_values[oi]], mode="markers", marker=dict(size=14, symbol="star", color=C.COLOR_NEXT, line=dict(color="white", width=1)),
                             name="Optimum der Gitterpunkte"), row=1, col=1)
    fig.update_xaxes(title_text=cfg["x"])
    fig.update_yaxes(title_text=cfg["y"], col=1)
    fig.update_annotations(font_size=12)
    return _fixed(fig, height=380)


def build_convergence_chart(a):
    """Wert der Empfehlung nach t Auswertungen, je Verfahren. Branin: Regret (logarithmisch); Terminal: wahre Kosten mit Optimum und Formel + Faustregel."""
    fig = go.Figure()
    ts = list(range(1, a.settings.n + 1))
    opt = a.optimum[2]
    for m in a.runs:
        if a.kind == "branin":
            y = [max(v - opt, 1e-3) for v in a.runs[m].truth_trace]
        else:
            y = [min(v, 3000) for v in a.runs[m].truth_trace]
        fig.add_trace(go.Scatter(x=ts, y=y, mode="lines", name=C.METHOD_LABELS[m], line=dict(color=C.METHOD_COLORS[m], width=2.5 if m.startswith("bo") else 1.8)))
    fig.update_xaxes(title_text="Auswertungen")
    if a.kind == "branin":
        fig.update_yaxes(title_text="Regret der Empfehlung (log)", type="log")
    else:
        fig.add_hline(y=opt, line_dash="dash", line_color=C.COLOR_NEXT, annotation_text="wahres Optimum", annotation_position="bottom right")
        fig.add_hline(y=a.rule[2], line_dash="dot", line_color=C.COLOR_BAD, annotation_text="Formel + Faustregel (0 Auswertungen)", annotation_position="top right")
        fig.update_yaxes(title_text="Wahre Kosten der Empfehlung (EUR/h)", range=[opt * 0.98, max(a.rule[2], opt * 1.25) * 1.05])
    return _fixed(fig, height=360)


def build_sweep_chart(curves, n_max, optimum_value, kind, rule_cost=None):
    """Median und mittlere Hälfte (25 bis 75 %) über das Budget (20 feste Läufe). Branin: Regret logarithmisch; Terminal: Kosten."""
    fig = go.Figure()
    ts = list(range(1, n_max + 1))

    def tr(v):
        return np.maximum(np.asarray(v[:n_max], float) - optimum_value, 1e-3) if kind == "branin" else np.asarray(v[:n_max], float)

    for m in curves:
        c = curves[m]
        col = C.METHOD_COLORS[m]
        fig.add_trace(go.Scatter(x=ts + ts[::-1], y=list(tr(c["q75"])) + list(tr(c["q25"]))[::-1], fill="toself", fillcolor=col, opacity=0.12,
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=ts, y=tr(c["median"]), mode="lines", name=C.METHOD_LABELS[m], line=dict(color=col, width=2.5)))
    fig.update_xaxes(title_text="Auswertungen")
    if kind == "branin":
        fig.update_yaxes(title_text="Regret, Median und 25 bis 75 % (log)", type="log")
    else:
        fig.add_hline(y=optimum_value, line_dash="dash", line_color=C.COLOR_NEXT, annotation_text="wahres Optimum", annotation_position="bottom right")
        if rule_cost is not None:
            fig.add_hline(y=rule_cost, line_dash="dot", line_color=C.COLOR_BAD, annotation_text="Formel + Faustregel (0 Auswertungen)", annotation_position="top right")
        fig.update_yaxes(title_text="Wahre Kosten der Empfehlung, Median und 25 bis 75 % (EUR/h)", range=[optimum_value * 0.98, optimum_value * 1.3])
    return _fixed(fig, height=380)


def build_calibration_chart(summary):
    ns = sorted(summary)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(n) for n in ns], y=[summary[n]["coverage_median"] for n in ns], name="Median über 20 Läufe", marker_color=C.COLOR_OBS))
    fig.add_trace(go.Scatter(x=[str(n) for n in ns], y=[summary[n]["coverage_q10"] for n in ns], mode="markers", name="schlechtester Lauf (10-%-Quantil)",
                             marker=dict(color=C.COLOR_BAD, size=10, symbol="diamond")))
    fig.add_hline(y=0.95, line_dash="dash", line_color=C.COLOR_NEXT, annotation_text="nominal 95 %", annotation_position="top left")
    fig.update_xaxes(title_text="Beobachtungen")
    fig.update_yaxes(title_text="Anteil der Gitterpunkte im 95-%-Intervall", range=[0, 1.05], tickformat=".0%")
    return _fixed(fig, height=330)


def build_full_chart(summary, rule):
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Bar(x=[f"{b} Auswertungen" for b in C.STUDY_BUDGETS], y=[summary[m][b]["median"] for b in C.STUDY_BUDGETS], name=C.METHOD_LABELS[m],
                             marker_color=C.METHOD_COLORS[m]))
    fig.add_hline(y=rule["long_cost"], line_dash="dot", line_color=C.COLOR_BAD, annotation_text="Formel + Faustregel (0 Auswertungen)", annotation_position="top left")
    lo = min(min(summary[m][b]["median"] for m in C.METHODS for b in C.STUDY_BUDGETS), rule["long_cost"])
    hi = max(summary[m][b]["median"] for m in C.METHODS for b in C.STUDY_BUDGETS)
    fig.update_yaxes(title_text="Wahre Kosten der Empfehlung, Median (EUR/h)", range=[lo * 0.9, hi * 1.05])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=360)


def build_bench_chart(summary):
    """Hartmann6: Regret (bester Wert minus bestes der Kandidatenmenge), Median je Verfahren und Budget, logarithmische Achse."""
    fig = go.Figure()
    for m in C.BENCH_METHODS:
        fig.add_trace(go.Bar(x=[f"{b} Auswertungen" for b in C.BENCH_BUDGETS], y=[max(summary[m][b]["median"], 1e-4) for b in C.BENCH_BUDGETS],
                             name=C.METHOD_LABELS[m], marker_color=C.METHOD_COLORS[m]))
    fig.update_yaxes(title_text="Regret, Median (log)", type="log")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=320)
