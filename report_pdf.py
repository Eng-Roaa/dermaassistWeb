"""
report_pdf.py  -  one-patient PDF report for the web app.

Ported from the Raspberry Pi project. Differences: images are read from the encrypted vault
through the service (nothing is written to disk) and the PDF is returned as bytes so the
browser can download it.

    pdf_bytes = generate_patient_report(svc, patient_id)

The report always carries a "not a diagnosis" banner and a footer on every page.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from clinical_summary import build_timeline, first_last_by_region, patient_summary

DISCLAIMER = ("This report was produced by an experimental research prototype. It is NOT a "
              "medical device and NOT a diagnosis. Classification results come from a small "
              "research model with six classes and no 'healthy skin' class, and image "
              "comparisons show pixel differences only. All findings must be reviewed by a "
              "qualified physician.")
FOOTER = "Experimental prototype - not a diagnosis"
CONTENT_WIDTH = 17 * cm                      # A4 (21 cm) minus 2 cm margins
MAX_SCANS_LISTED = 15
MAX_THUMBS = 12
MAX_COMPARISON_REGIONS = 3

_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]


def _register_fonts():
    """DejaVu Sans if installed (wide character coverage), else built-in Helvetica."""
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists() and Path(bold).exists():
            try:
                pdfmetrics.registerFont(TTFont("DermaSans", regular))
                pdfmetrics.registerFont(TTFont("DermaSans-Bold", bold))
                return "DermaSans", "DermaSans-Bold", True
            except Exception:
                pass
    return "Helvetica", "Helvetica-Bold", False


class _Text:
    """Escapes user text for reportlab and removes characters the built-in font cannot draw."""

    def __init__(self, unicode_font: bool):
        self.unicode_font = unicode_font

    def __call__(self, value) -> str:
        text = "" if value is None else str(value)
        if not self.unicode_font:
            text = text.encode("cp1252", "replace").decode("cp1252")
        return escape(text).replace("\n", "<br/>")


def _styles(font, bold):
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName=bold, fontSize=20, spaceAfter=4),
        "h": ParagraphStyle("h", parent=base["Heading2"], fontName=bold, fontSize=13, spaceBefore=12,
                            spaceAfter=5, textColor=colors.HexColor("#0d3b66")),
        "body": ParagraphStyle("body", parent=base["Normal"], fontName=font, fontSize=9.5, leading=13),
        "small": ParagraphStyle("small", parent=base["Normal"], fontName=font, fontSize=8, leading=10,
                                textColor=colors.HexColor("#444444")),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontName=font, fontSize=8.5, leading=10.5),
        "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontName=bold, fontSize=8.5, leading=10.5,
                                textColor=colors.white),
        "banner": ParagraphStyle("banner", parent=base["Normal"], fontName=bold, fontSize=9.5, leading=13,
                                 textColor=colors.HexColor("#8a1c1c")),
    }


def _table(rows, col_widths, st, header=True):
    """rows: list of lists of strings already escaped by _Text."""
    data = []
    for i, row in enumerate(rows):
        style = st["cellb"] if (header and i == 0) else st["cell"]
        data.append([Paragraph(cell, style) for cell in row])
    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b7c3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d3b66")))
        for i in range(1, len(data)):
            if i % 2 == 0:
                commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f2f5f9")))
    table.setStyle(TableStyle(commands))
    return table


def _pil_flowable(pil_image, max_w, max_h):
    """A thumbnail of a PIL image as a reportlab Image, or None if it cannot be drawn."""
    try:
        im = pil_image.convert("RGB")
        im.thumbnail((600, 600))
        w_px, h_px = im.size
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    except Exception:
        return None
    buf.seek(0)
    scale = min(max_w / w_px, max_h / h_px)
    return Image(buf, width=w_px * scale, height=h_px * scale)


def generate_patient_report(svc, patient_id: int, now: datetime | None = None) -> bytes:
    """Build the report and return the PDF file as bytes. Permission checks are done by the caller."""
    now = now or datetime.now()
    summary = patient_summary(svc.db, patient_id)
    patient = summary["patient"]
    code = patient["patient_code"]

    font, bold, unicode_font = _register_fonts()
    t = _Text(unicode_font)
    st = _styles(font, bold)
    story = []

    # ---- title + banner
    story.append(Paragraph("DermaAssist - Patient Skin Report", st["title"]))
    banner = Table([[Paragraph("NOT A DIAGNOSIS. Experimental research prototype; results must be "
                               "reviewed by a qualified physician.", st["banner"])]],
                   colWidths=[CONTENT_WIDTH])
    banner.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#8a1c1c")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdeeee")),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [banner, Spacer(1, 8)]

    # ---- 1. patient
    story.append(Paragraph("1. Patient", st["h"]))
    age = summary["age"]
    rows = [
        ["<b>Patient ID</b>", t(code), "<b>Report created</b>", t(f"{now:%Y-%m-%d %H:%M}")],
        ["<b>Name</b>", t(patient["full_name"]), "<b>Registered</b>", t(patient["created_at"])],
        ["<b>Date of birth</b>", t(patient["date_of_birth"] or "-"), "<b>Age</b>",
         t(age if age is not None else "-")],
        ["<b>Sex</b>", t(patient["sex"] or "-"), "<b>Phone</b>", t(patient["phone"] or "-")],
    ]
    story.append(_table(rows, [3 * cm, 5.5 * cm, 3 * cm, 5.5 * cm], st, header=False))
    if patient["notes"]:
        story += [Spacer(1, 4), Paragraph("Notes: " + t(patient["notes"]), st["body"])]
    first, last = summary["first_visit"], summary["last_visit"]
    story += [Spacer(1, 6), Paragraph(
        f"<b>Summary:</b> {summary['scan_count']} AI scan(s), {summary['visit_count']} tracking visit(s)"
        + (f" ({t(first['visit_date'][:10])} to {t(last['visit_date'][:10])})" if first else "")
        + f", {summary['treatment_count']} treatment record(s).", st["body"])]

    # ---- 2. AI classification
    story.append(Paragraph("2. AI classification results", st["h"]))
    if not summary["analysed_count"]:
        story.append(Paragraph("No classified images for this patient.", st["body"]))
    else:
        counts = ", ".join(f"{t(k)}: {v}" for k, v in summary["class_counts"].items())
        story.append(Paragraph(
            f"{summary['analysed_count']} classified image(s). Confident results by class: "
            f"{counts or 'none'}. Uncertain (low confidence): {summary['uncertain_count']}.", st["body"]))
        story.append(Spacer(1, 4))
        rows = [["Date", "Result", "Confidence", "Model"]]
        analysed = [s for s in summary["scans"] if s["predicted_class"]]
        for s in analysed[:MAX_SCANS_LISTED]:
            result = (f"Uncertain (top score: {t(s['predicted_class'])})"
                      if s["is_uncertain"] else t(s["predicted_class"]))
            rows.append([t(s["captured_at"]), result, f"{s['confidence'] * 100:.1f}%", t(s["model_name"])])
        story.append(_table(rows, [4 * cm, 6.5 * cm, 2.5 * cm, 4 * cm], st))
        if len(analysed) > MAX_SCANS_LISTED:
            story.append(Paragraph(f"Showing the latest {MAX_SCANS_LISTED} of {len(analysed)} results.",
                                   st["small"]))

    # ---- 3. tracking visits
    story.append(Paragraph("3. Skin tracking visits", st["h"]))
    visits = summary["visits"]
    if not visits:
        story.append(Paragraph("No tracking visits recorded.", st["body"]))
    else:
        rows = [["#", "Date", "Region", "ROI", "Notes"]]
        for v in visits:
            roi = f"{v['roi_w']}x{v['roi_h']} px" if v["roi_w"] else "whole image"
            rows.append([f"{v['visit_number']}", t(v["visit_date"]), t(v["skin_region"]), roi,
                         t(v["notes"] or "")])
        story.append(_table(rows, [1 * cm, 4 * cm, 3 * cm, 3 * cm, 6 * cm], st))

        cells = []
        for v in visits[-MAX_THUMBS:]:
            try:
                flow = _pil_flowable(svc.visit_image(v, "roi"), 5 * cm, 4 * cm)
            except Exception:
                flow = None
            caption = Paragraph(f"V{v['visit_number']} - {t(v['visit_date'][:10])}", st["small"])
            cells.append([flow or Paragraph("(image unavailable)", st["small"]), caption])
        story.append(Spacer(1, 6))
        grid_rows = []
        for i in range(0, len(cells), 3):
            chunk = cells[i:i + 3]
            while len(chunk) < 3:
                chunk.append(["", ""])
            grid_rows.append([c[0] for c in chunk])
            grid_rows.append([c[1] for c in chunk])
        grid = Table(grid_rows, colWidths=[CONTENT_WIDTH / 3] * 3)
        grid.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(grid)
        if len(visits) > MAX_THUMBS:
            story.append(Paragraph(f"Showing the latest {MAX_THUMBS} of {len(visits)} visits.", st["small"]))

    # ---- 4. before / after
    story.append(Paragraph("4. Before / After comparison", st["h"]))
    pairs = first_last_by_region(visits)
    if not pairs:
        story.append(Paragraph("A comparison needs at least two visits for the same skin region.",
                               st["body"]))
    else:
        for region, (v1, v2) in list(pairs.items())[:MAX_COMPARISON_REGIONS]:
            part = [Paragraph(
                f"<b>{t(region)}</b>: first visit #{v1['visit_number']} ({t(v1['visit_date'][:10])}) vs "
                f"latest visit #{v2['visit_number']} ({t(v2['visit_date'][:10])})", st["body"])]
            try:
                result, sheet = svc._compare(v1, v2)            # sheet: RGB array
                buf = io.BytesIO()
                PILImage.fromarray(sheet).save(buf, "PNG")
                buf.seek(0)
                height, width = sheet.shape[:2]
                part += [Spacer(1, 3),
                         Image(buf, width=CONTENT_WIDTH, height=CONTENT_WIDTH * height / width)]
                line = (f"Pixel difference: mean colour distance {result.mean_delta_e:.1f}; clearly "
                        f"different pixels {result.changed_percent:.1f}%; alignment "
                        f"{'applied' if result.aligned else 'not applied'}.")
                if result.warnings:
                    line += " " + " ".join(result.warnings)
                part.append(Paragraph(t(line), st["small"]))
            except Exception as exc:
                part.append(Paragraph(t(f"Comparison unavailable: {exc}"), st["small"]))
            part.append(Spacer(1, 6))
            story.append(KeepTogether(part))
        story.append(Paragraph(
            "The difference map shows pixel differences only. It is not evidence of improvement or "
            "worsening; lighting, angle and distance affect it.", st["small"]))

    # ---- 5. treatments
    story.append(Paragraph("5. Treatment history", st["h"]))
    treatments = summary["treatments"]
    if not treatments:
        story.append(Paragraph("No treatments recorded.", st["body"]))
    else:
        by_id = {v["id"]: v for v in visits}
        rows = [["Treatment", "Started", "Stopped", "Linked visit", "Notes"]]
        for tr in treatments:
            linked = by_id.get(tr["visit_id"])
            rows.append([t(tr["treatment_name"]), t(tr["start_date"]), t(tr["end_date"] or "ongoing"),
                         f"#{linked['visit_number']}" if linked else "-", t(tr["notes"] or "")])
        story.append(_table(rows, [4 * cm, 2.5 * cm, 2.5 * cm, 2 * cm, 6 * cm], st))

    # ---- 6. timeline
    story.append(Paragraph("6. Timeline: visits and treatment", st["h"]))
    timeline = build_timeline(visits, treatments)
    if not timeline:
        story.append(Paragraph("Nothing to show yet.", st["body"]))
    else:
        rows = [["Date", "Event"]] + [[t(r["date"]), t(r["text"])] for r in timeline]
        story.append(_table(rows, [3 * cm, 14 * cm], st))

    story += [Spacer(1, 14), Paragraph("Important notice", st["h"]), Paragraph(t(DISCLAIMER), st["body"])]

    # ---- build
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 7.5)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(2 * cm, 1.2 * cm, f"{FOOTER}  |  Patient {code}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title=f"Skin report {code}", author="DermaAssist")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()
