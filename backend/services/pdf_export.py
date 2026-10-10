import io
import logging
from html import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    KeepTogether,
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import inch

logger = logging.getLogger("ats_resume_scorer")


def _clean_text(value):
    if value is None:
        return ""

    if isinstance(value, (dict, list, tuple)):
        value = str(value)

    return escape(str(value)).replace("\n", "<br/>")


def _add_html_content(story, html_content, styles):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content or "", "html.parser")

    for element in soup.find_all(
        ["script", "style", "nav", "button", "input", "svg"]
    ):
        element.decompose()

    body = soup.body or soup
    for element in body.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "div", "table", "pre"]
    ):
        if element.name == "table":
            rows = []
            for row in element.find_all("tr"):
                cells = [
                    cell.get_text(" ", strip=True)
                    for cell in row.find_all(["th", "td"])
                ]
                if cells:
                    rows.append(" | ".join(cells))

            if rows:
                story.append(
                    Paragraph(
                        "<br/>".join(_clean_text(row) for row in rows),
                        styles["BodyText"],
                    )
                )
                story.append(Spacer(1, 8))

        elif element.name == "li":
            text = element.get_text(" ", strip=True)
            if text:
                story.append(
                    Paragraph(
                        "• " + _clean_text(text),
                        styles["BodyText"],
                    )
                )
                story.append(Spacer(1, 4))

        elif element.name in ("h1", "h2"):
            text = element.get_text(" ", strip=True)
            if text:
                story.append(
                    Paragraph(_clean_text(text), styles["Heading1"])
                )
                story.append(Spacer(1, 6))

        elif element.name in ("h3", "h4"):
            text = element.get_text(" ", strip=True)
            if text:
                story.append(
                    Paragraph(_clean_text(text), styles["Heading2"])
                )
                story.append(Spacer(1, 4))

        elif element.name in ("p", "pre"):
            text = element.get_text(" ", strip=True)
            if text:
                story.append(
                    Paragraph(_clean_text(text), styles["BodyText"])
                )
                story.append(Spacer(1, 6))


def generate_combined_pdf(html_docs: dict[str, str]) -> bytes:
    if not html_docs:
        raise ValueError("No HTML content was provided for PDF generation.")

    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="ATS Resume Analysis Report",
    )

    styles = getSampleStyleSheet()
    styles["Title"].alignment = TA_CENTER
    styles["Title"].textColor = colors.HexColor("#243B53")
    styles["Heading1"].textColor = colors.HexColor("#243B53")
    styles["Heading2"].textColor = colors.HexColor("#334E68")
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 4

    story = [
        Paragraph("ATS Resume Analysis Report", styles["Title"]),
        Spacer(1, 18),
    ]

    for index, (name, html_content) in enumerate(html_docs.items()):
        if index > 0:
            story.append(PageBreak())

        story.append(
            Paragraph(_clean_text(name.replace("_", " ").title()), styles["Heading1"])
        )
        story.append(Spacer(1, 10))

        _add_html_content(story, html_content, styles)

    document.build(story)
    return output.getvalue()