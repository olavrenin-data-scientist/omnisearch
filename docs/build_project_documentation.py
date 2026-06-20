"""
Builds docs/PROJECT_DOCUMENTATION.pdf from docs/PROJECT_DOCUMENTATION.md.

Unlike docs/build_walkthrough.py (which embeds its own authored markdown
string), this script *reads* the existing Markdown file and renders it to a
styled PDF. It supports the markdown the doc actually uses: headings, lists,
fenced code blocks (syntax-highlighted via Pygments), GitHub-style pipe
tables, horizontal rules, and inline **bold** / *italic* / `code` / [links].

Pure-Python via ReportLab — no Pango/Cairo/LaTeX/pandoc system deps, so it
runs anywhere `pip install reportlab pygments` works.

Run:
    .venv/bin/python docs/build_project_documentation.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from pygments import lex
from pygments.lexers import (
    BashLexer,
    CssLexer,
    HtmlLexer,
    JavascriptLexer,
    PythonLexer,
    TextLexer,
)
from pygments.styles import get_style_by_name

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)

ROOT = Path(__file__).resolve().parent.parent
SRC_MD = ROOT / "docs" / "PROJECT_DOCUMENTATION.md"
OUT_PDF = ROOT / "docs" / "PROJECT_DOCUMENTATION.pdf"
TITLE = "OmniSearch — Project Documentation"


# ----------------------------------------------------------------------
# Fonts — DejaVu (shipped with matplotlib) gives full Unicode coverage
# (box-drawing, arrows, ✓, ±, ×, →) that ReportLab's built-in Type-1
# fonts lack. Falls back to Helvetica/Courier if DejaVu isn't importable.
# ----------------------------------------------------------------------
BODY_FONT = "Helvetica"
BODY_BOLD = "Helvetica-Bold"
BODY_ITALIC = "Helvetica-Oblique"
MONO_FONT = "Courier"
MONO_BOLD = "Courier-Bold"


def _register_fonts() -> None:
    global BODY_FONT, BODY_BOLD, BODY_ITALIC, MONO_FONT, MONO_BOLD
    try:
        import matplotlib
    except ImportError:
        return
    ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    fonts = {
        "DejaVuSans": ttf / "DejaVuSans.ttf",
        "DejaVuSans-Bold": ttf / "DejaVuSans-Bold.ttf",
        "DejaVuSans-Oblique": ttf / "DejaVuSans-Oblique.ttf",
        "DejaVuSansMono": ttf / "DejaVuSansMono.ttf",
        "DejaVuSansMono-Bold": ttf / "DejaVuSansMono-Bold.ttf",
    }
    if not all(p.exists() for p in fonts.values()):
        return
    for name, path in fonts.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
        italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-Bold",
    )
    BODY_FONT, BODY_BOLD, BODY_ITALIC = "DejaVuSans", "DejaVuSans-Bold", "DejaVuSans-Oblique"
    MONO_FONT, MONO_BOLD = "DejaVuSansMono", "DejaVuSansMono-Bold"


# ----------------------------------------------------------------------
# Palette + inline/code formatting helpers
# ----------------------------------------------------------------------
TEAL = colors.HexColor("#14b8a6")
NAVY = colors.HexColor("#1a3a8a")
DEEP_TEAL = colors.HexColor("#0f4f48")
INK = colors.HexColor("#1a1f2c")
CODE_BG = colors.HexColor("#f5f7fa")
CODE_BORDER = colors.HexColor("#d8dce5")
TABLE_HEAD_BG = colors.HexColor("#1a3a8a")
TABLE_ROW_ALT = colors.HexColor("#eef1f6")
LINK_COLOR = colors.HexColor("#1c6fb8")

_LEXERS = {
    "python": PythonLexer, "py": PythonLexer,
    "javascript": JavascriptLexer, "js": JavascriptLexer, "jsx": JavascriptLexer,
    "html": HtmlLexer, "css": CssLexer,
    "bash": BashLexer, "sh": BashLexer, "shell": BashLexer, "zsh": BashLexer,
}
_PYG_STYLE = get_style_by_name("default")


def _inline(text: str) -> str:
    """Convert a markdown span to ReportLab mini-markup.

    Handles escaping, `code`, [links](url), **bold**, and *italic*.
    """
    text = html.escape(text, quote=False)
    spans: list[str] = []

    def _stash(m: "re.Match") -> str:
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    # Stash inline code first so its contents are not touched by later passes.
    text = re.sub(r"`([^`]+)`", _stash, text)

    # Markdown links: [label](target). External http(s) targets become real
    # clickable links; internal anchors (#section) render as plain label text
    # since intra-PDF anchors aren't wired up.
    def _link(m: "re.Match") -> str:
        label, target = m.group(1), m.group(2)
        if target.startswith(("http://", "https://")):
            return f'<link href="{target}"><font color="#1c6fb8"><u>{label}</u></font></link>'
        return label

    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", _link, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)

    def _restore(m: "re.Match") -> str:
        code = spans[int(m.group(1))]
        return f'<font face="{MONO_FONT}" size="9" color="#b5076b">{code}</font>'

    return re.sub(r"\x00(\d+)\x00", _restore, text)


def _highlight(code: str, lang: str) -> str:
    lexer_cls = _LEXERS.get((lang or "").lower(), TextLexer)
    out = []
    for ttype, value in lex(code, lexer_cls()):
        esc = html.escape(value, quote=False)
        st = _PYG_STYLE.style_for_token(ttype)
        prefix, suffix = "", ""
        if st.get("color"):
            prefix += f'<font color="#{st["color"]}">'
            suffix = "</font>" + suffix
        if st.get("bold"):
            prefix += "<b>"
            suffix = "</b>" + suffix
        if st.get("italic"):
            prefix += "<i>"
            suffix = "</i>" + suffix
        out.append(prefix + esc + suffix)
    return "".join(out)


# ----------------------------------------------------------------------
# Paragraph styles
# ----------------------------------------------------------------------
def _styles() -> dict:
    base = getSampleStyleSheet()["Normal"]
    body = ParagraphStyle(
        "Body", parent=base, fontName=BODY_FONT, fontSize=10.5, leading=15.5,
        textColor=INK, spaceBefore=2, spaceAfter=7, alignment=TA_LEFT,
    )
    return {
        "body": body,
        "list": ParagraphStyle("List", parent=body, leftIndent=16, spaceAfter=4),
        "h1": ParagraphStyle("H1", parent=body, fontName=BODY_BOLD, fontSize=21,
                             leading=25, textColor=colors.HexColor("#0e1118"),
                             spaceBefore=10, spaceAfter=4),
        "h2": ParagraphStyle("H2", parent=body, fontName=BODY_BOLD, fontSize=15,
                             leading=19, textColor=NAVY, spaceBefore=16, spaceAfter=3),
        "h3": ParagraphStyle("H3", parent=body, fontName=BODY_BOLD, fontSize=12,
                             leading=15, textColor=DEEP_TEAL, spaceBefore=12, spaceAfter=2),
        "h4": ParagraphStyle("H4", parent=body, fontName=BODY_BOLD, fontSize=10.8,
                             leading=14, textColor=colors.HexColor("#444444"),
                             spaceBefore=9, spaceAfter=1),
        "quote": ParagraphStyle("Quote", parent=body, leftIndent=12,
                                textColor=colors.HexColor("#475067"),
                                fontName=BODY_ITALIC, borderColor=TEAL,
                                spaceBefore=4, spaceAfter=8),
        "code": ParagraphStyle("Code", parent=base, fontName=MONO_FONT, fontSize=8.0,
                               leading=10.8, textColor=INK, backColor=CODE_BG,
                               borderColor=CODE_BORDER, borderWidth=0.5,
                               borderPadding=(6, 6, 6, 6), leftIndent=2, rightIndent=2,
                               spaceBefore=4, spaceAfter=9,
                               wordWrap="CJK", splitLongWords=1,
                               allowWidows=1, allowOrphans=1),
        "th": ParagraphStyle("TH", parent=body, fontName=BODY_BOLD, fontSize=9,
                             leading=12, textColor=colors.white, spaceBefore=0, spaceAfter=0),
        "td": ParagraphStyle("TD", parent=body, fontSize=9, leading=12,
                             spaceBefore=0, spaceAfter=0),
    }


# ----------------------------------------------------------------------
# Table parsing / rendering
# ----------------------------------------------------------------------
def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    # Split on unescaped pipes.
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)]


def _is_table_separator(line: str) -> bool:
    return bool(re.match(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$", line))


def _make_table(header: list[str], rows: list[list[str]], styles: dict, avail_width: float) -> Table:
    n = len(header)
    data = [[Paragraph(_inline(c), styles["th"]) for c in header]]
    for r in rows:
        cells = (r + [""] * n)[:n]
        data.append([Paragraph(_inline(c), styles["td"]) for c in cells])

    col_width = avail_width / n
    table = Table(data, colWidths=[col_width] * n, repeatRows=1, hAlign="LEFT")
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, CODE_BORDER),
    ]
    for ri in range(1, len(data)):
        if ri % 2 == 0:
            ts.append(("BACKGROUND", (0, ri), (-1, ri), TABLE_ROW_ALT))
    table.setStyle(TableStyle(ts))
    return table


# ----------------------------------------------------------------------
# Markdown → flowables
# ----------------------------------------------------------------------
def _build_flowables(md: str, styles: dict, avail_width: float) -> list:
    flow: list = []
    lines = md.split("\n")
    i, n = 0, len(lines)
    para: list[str] = []

    def flush_para() -> None:
        if not para:
            return
        text = " ".join(s.strip() for s in para).strip()
        para.clear()
        if not text:
            return
        m = re.match(r"^(\d+\.|[-*])\s+(.*)$", text)
        if m:
            bullet = "•" if m.group(1) in ("-", "*") else m.group(1)
            flow.append(Paragraph(f"{bullet}&nbsp;&nbsp;{_inline(m.group(2))}",
                                  styles["list"]))
        else:
            flow.append(Paragraph(_inline(text), styles["body"]))

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Fenced code block
        if stripped.startswith("```"):
            flush_para()
            lang = stripped[3:].strip()
            i += 1
            block: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code = "\n".join(block)
            flow.append(XPreformatted(_highlight(code, lang), styles["code"]))
            continue

        # Pipe table: a header row followed by a separator row
        if "|" in line and i + 1 < n and _is_table_separator(lines[i + 1]):
            flush_para()
            header = _split_row(line)
            i += 2  # skip header + separator
            rows: list[list[str]] = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            flow.append(_make_table(header, rows, styles, avail_width))
            flow.append(Spacer(1, 8))
            continue

        # Horizontal rule
        if re.match(r"^\s*([-*_])\1{2,}\s*$", line):
            flush_para()
            flow.append(HRFlowable(width="100%", thickness=0.6,
                                   color=colors.HexColor("#c8d0e0"),
                                   spaceBefore=4, spaceAfter=8))
            i += 1
            continue

        # Headings
        if stripped.startswith("#"):
            flush_para()
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            key = {1: "h1", 2: "h2", 3: "h3"}.get(level, "h4")
            flow.append(Paragraph(_inline(text), styles[key]))
            if level == 1:
                flow.append(HRFlowable(width="100%", thickness=2, color=TEAL,
                                       spaceBefore=2, spaceAfter=8))
            elif level == 2:
                flow.append(HRFlowable(width="100%", thickness=0.6,
                                       color=colors.HexColor("#c8d0e0"),
                                       spaceBefore=1, spaceAfter=6))
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            flush_para()
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            flow.append(Paragraph(_inline(" ".join(quote_lines)), styles["quote"]))
            continue

        if stripped == "":
            flush_para()
            i += 1
            continue

        # A new list item starts its own paragraph even without a blank line.
        if re.match(r"^\s*(\d+\.|[-*])\s+", line) and para:
            flush_para()

        para.append(line)
        i += 1

    flush_para()
    return flow


# ----------------------------------------------------------------------
# Page furniture: running header/footer with page number
# ----------------------------------------------------------------------
class DocTemplate(BaseDocTemplate):
    def __init__(self, path: str, **kw):
        super().__init__(path, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="main")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                            onPage=self._decorate)])

    def _decorate(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(BODY_FONT, 8.5)
        canvas.setFillColor(colors.HexColor("#8b94a8"))
        canvas.drawString(doc.leftMargin, 1.1 * cm, TITLE)
        canvas.drawRightString(doc.leftMargin + doc.width, 1.1 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#d8dce5"))
        canvas.line(doc.leftMargin, 1.45 * cm, doc.leftMargin + doc.width, 1.45 * cm)
        canvas.restoreState()


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def build_pdf(src_md: Path, out_pdf: Path, header_title: str, subtitle: str) -> None:
    """Render a Markdown file to a styled PDF.

    ``header_title`` is the running header/footer label; ``subtitle`` is the
    teal line under the big "OmniSearch" title block.
    """
    global TITLE
    TITLE = header_title  # picked up by DocTemplate._decorate via module global
    _register_fonts()
    styles = _styles()

    md = src_md.read_text(encoding="utf-8")
    # Drop the leading H1 — we render a dedicated title block instead.
    md = re.sub(r"\A\s*#\s+.*\n", "", md, count=1)

    left = right = 1.9 * cm
    avail_width = A4[0] - left - right

    flow: list = [
        Paragraph("OmniSearch", ParagraphStyle(
            "Title", fontName=BODY_BOLD, fontSize=30, leading=34,
            textColor=colors.HexColor("#0e1118"), spaceAfter=2)),
        Paragraph(subtitle, ParagraphStyle(
            "Sub", fontName=BODY_FONT, fontSize=16, leading=20,
            textColor=TEAL, spaceAfter=10)),
        HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=12),
    ]
    flow += _build_flowables(md, styles, avail_width)

    doc = DocTemplate(
        str(out_pdf), pagesize=A4,
        leftMargin=left, rightMargin=right,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title=header_title, author="OmniSearch",
    )
    doc.build(flow)
    try:
        shown = out_pdf.resolve().relative_to(ROOT)
    except ValueError:
        shown = out_pdf
    print(f"wrote {shown}  ({out_pdf.stat().st_size:_} bytes)")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Render a project Markdown doc to a styled PDF.")
    p.add_argument("--src", default=str(SRC_MD), help="Source Markdown file.")
    p.add_argument("--out", default=str(OUT_PDF), help="Output PDF path.")
    p.add_argument("--title", default=TITLE, help="Running header/footer title.")
    p.add_argument("--subtitle", default="Project Documentation",
                   help="Subtitle under the title block.")
    args = p.parse_args()

    build_pdf(Path(args.src), Path(args.out), args.title, args.subtitle)


if __name__ == "__main__":
    main()
