"""PDF-Export der aktuellen Auswertung (fpdf2, Kernschriften; Umlaute sind dort erlaubt, Gedankenstrich und Eurozeichen nicht)."""

from fpdf import FPDF

import bao_constants as C
import bao_evaluation as ev


def _num(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def build_pdf(a):
    """Einseitiger Bericht: Einstellungen und Kosten der Empfehlung je Verfahren. Gibt die Bytes der PDF-Datei zurück."""
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Bayessche Optimierung am Terminal", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sebastian Hanisch, Operations Research und Machine Learning", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for k, v in (("Auswertungen", str(s.n)), ("Lkw je Auswertung", _num(s.trucks)), ("Zufalls-Seed", str(s.seed))):
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Wahre Kosten der Empfehlung (EUR je Stunde)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    opt = a.optimum[2]
    for m in C.METHODS:
        c = a.runs[m].truth_trace[-1]
        pdf.cell(95, 6, C.METHOD_LABELS[m])
        pdf.cell(0, 6, f"{_num(c)} ({_num(ev.regret_pct(c, opt), 1)} % über dem Optimum)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(95, 6, "Formel ohne Platzgrenze (0 Auswertungen)")
    pdf.cell(0, 6, f"{_num(a.formula[2])} ({_num(ev.regret_pct(a.formula[2], opt), 1)} % über dem Optimum)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(95, 6, "Wahres Optimum der Scheibe")
    pdf.cell(0, 6, _num(opt), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Auswertungen stammen aus einer Tabelle echter Simulationen; die Zahlen gelten für dieses Netz, diese Kosten und diese Einstellungen.")
    return bytes(pdf.output())
