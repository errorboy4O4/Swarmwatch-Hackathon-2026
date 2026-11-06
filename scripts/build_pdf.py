"""Build the submission PDF from report/report_final.md (template-style layout).
Usage: python scripts/build_pdf.py"""
import re, os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether, PageBreak

SRC, OUT = "report/report_final.md", "report/SwarmWatch_report.pdf"
BODY = ParagraphStyle("b", fontName="Times-Roman", fontSize=10.3, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5)
H2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#333333"))
CAP = ParagraphStyle("cap", parent=BODY, fontName="Times-Italic", fontSize=9.2, leading=11.5, alignment=TA_JUSTIFY)
TT = ParagraphStyle("tt", fontName="Times-Bold", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=6)
CTR = ParagraphStyle("c", parent=BODY, alignment=TA_CENTER, spaceAfter=2)
ABS = ParagraphStyle("abs", parent=BODY, fontName="Times-Italic", leftIndent=28, rightIndent=28)
CELL = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.6, leading=10.4)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Times-Bold")

def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r'<font name="Courier" size="9">\1</font>', t)
    t = re.sub(r"(https?://[^\s)]+[^\s).,;])", r'<font color="#1a4fa0">\1</font>', t)
    return t

lines = open(SRC, encoding="utf-8").read().split("\n")
story, meta, i = [], {}, 0
def footer(c, d):
    c.saveState(); c.setFont("Times-Roman", 8.5); c.drawCentredString(letter[0] / 2, 0.5 * inch, str(d.page)); c.restoreState()
while i < len(lines):
    ln = lines[i]
    if not ln.strip(): i += 1; continue
    if ln.startswith("# "):
        story += [HRFlowable(width="100%", thickness=3, color=colors.black, spaceAfter=10), Paragraph(inline(ln[2:]) + "<super>1</super>", TT),
                  HRFlowable(width="100%", thickness=0.8, color=colors.black, spaceAfter=8)]
    elif ln.startswith("AUTHOR:"):
        n, a = [x.strip() for x in ln[7:].split("|")]
        story += [Paragraph(f"<b>{inline(n)}</b>", CTR), Paragraph(inline(a), CTR), Spacer(1, 4)]
    elif ln.startswith("WITH:"):
        story += [Paragraph("<b>With</b>", CTR), Paragraph(inline(ln[5:].strip()), CTR)]
    elif ln.startswith("FOOT:"): meta["foot"] = ln[5:].strip()
    elif ln.startswith("## "):
        h = ln[3:]
        if h.startswith("Appendix A") : story.append(PageBreak())
        story.append(Paragraph(f"<b>{inline(h)}</b>" if h == "Abstract" else inline(h), ParagraphStyle("h", parent=H2, alignment=TA_CENTER) if h == "Abstract" else H2))
    elif ln.startswith("!["):
        m = re.match(r"!\[(.*)\]\((.*)\)", ln); cap, path = m.group(1), m.group(2)
        from PIL import Image as PI
        w, h = PI.open(path.replace("report/", "report/", 1)).size; W = 5.6 * inch
        story.append(KeepTogether([Image(path, width=W, height=W * h / w), Spacer(1, 3), Paragraph(inline(cap), CAP), Spacer(1, 6)]))
    elif ln.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            if not re.match(r"^\|[-| ]+\|$", lines[i]): rows.append([c.strip() for c in lines[i].strip("|").split("|")])
            i += 1
        data = [[Paragraph(inline(c), CELLB if r == 0 else CELL) for c in row] for r, row in enumerate(rows)]
        ncol = len(rows[0]); W = 6.3 * inch; first = W * (0.46 if ncol == 4 else 0.52 if ncol == 3 else 0.30 if ncol >= 6 else 0.5)
        cw = [first] + [(W - first) / (ncol - 1)] * (ncol - 1)
        if ncol == 7: cw = [W * x for x in (0.10, 0.25, 0.12, 0.12, 0.14, 0.14, 0.13)]
        t = Table(data, colWidths=cw, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        blk = [t, Spacer(1, 6)]
        if story and isinstance(story[-1], Paragraph) and story[-1].text.startswith("Table "): blk = [story.pop()] + blk
        story.append(KeepTogether(blk)); continue
    elif re.match(r"^(- |\d+\. )", ln):
        items = []
        while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i]):
            items.append(lines[i]); i += 1
        for it in items:
            num = re.match(r"^(\d+)\. ", it)
            bullet, txt = (num.group(1) + ".", it[num.end():]) if num else ("\u2022", it[2:])
            story.append(Paragraph(f"{bullet}&nbsp;&nbsp;{inline(txt)}", ParagraphStyle("li", parent=BODY, leftIndent=18, firstLineIndent=-12, spaceAfter=3)))
        continue
    else:
        was_abs = bool(story and isinstance(story[-1], Paragraph) and story[-1].text.startswith("<b>Abstract"))
        story.append(Paragraph(inline(ln), ABS if (story and isinstance(story[-1], Paragraph) and story[-1].text.startswith("<b>Abstract")) else
                               CAP if ln.startswith("Table ") else BODY))
        if was_abs:
            story += [Spacer(1, 6), HRFlowable(width="35%", thickness=0.5, color=colors.grey),
                      Paragraph("<super>1</super> " + inline(meta.get("foot", "")), ParagraphStyle("f", parent=BODY, fontSize=8.5, leading=10.5, alignment=0))]
    i += 1
doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.95 * inch, rightMargin=0.95 * inch, topMargin=0.9 * inch, bottomMargin=0.85 * inch,
                        title="What a Monitor Still Sees When It Cannot Read the Messages", author="Kaushik Gaur")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT, os.path.getsize(OUT))
