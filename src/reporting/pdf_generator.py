"""
src/reporting/pdf_generator.py
-------------------------------
Production PDF Report Generator for LLM Response Evaluation Framework.
Generates an executive, professional PDF evaluation report adhering to:
(1) Cover / Summary page with batch metadata, overall stats KPI tiles, verdict distribution
    pie chart, and dimensional average score horizontal bar chart.
(2) Actionable Recommendations page with empirical metric triggers, engineering remediation advice,
    and priority badges.
(3) Per-Response detail sections with color-coded verdict badges, dimension breakdown tables,
    diagnostic reasoning, flagged hallucinated claims, and missing aspects.
Includes running header/footer with "Page X of Y" pagination, consistent typography,
and clean text wrapping without overflowing or clipping.
"""

import io
import html
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics.charts.piecharts import Pie

from src.reporting.schemas import (
    BatchReportData,
    RecommendationItem,
    PerResponseReportDetail,
    OverallStats,
    HallucinationFrequency,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design System Color Tokens (Matching Dashboard UI)
# ---------------------------------------------------------------------------
COLOR_PRIMARY = colors.HexColor("#1d4ed8")        # Brand blue accent
COLOR_PRIMARY_LIGHT = colors.HexColor("#eff6ff")
COLOR_PRIMARY_BORDER = colors.HexColor("#bfdbfe")
COLOR_PRIMARY_TEXT = colors.HexColor("#1e40af")

COLOR_PASS = colors.HexColor("#047857")           # Emerald green
COLOR_PASS_LIGHT = colors.HexColor("#ecfdf5")
COLOR_PASS_BORDER = colors.HexColor("#a7f3d0")
COLOR_PASS_TEXT = colors.HexColor("#065f46")

COLOR_NEEDS = colors.HexColor("#b45309")          # Warning amber
COLOR_NEEDS_LIGHT = colors.HexColor("#fffbeb")
COLOR_NEEDS_BORDER = colors.HexColor("#fde68a")
COLOR_NEEDS_TEXT = colors.HexColor("#92400e")

COLOR_FAIL = colors.HexColor("#b91c1c")           # Danger red
COLOR_FAIL_LIGHT = colors.HexColor("#fef2f2")
COLOR_FAIL_BORDER = colors.HexColor("#fecaca")
COLOR_FAIL_TEXT = colors.HexColor("#991b1b")

COLOR_SLATE_900 = colors.HexColor("#0f172a")      # Headings
COLOR_SLATE_800 = colors.HexColor("#1e293b")      # Subtitles
COLOR_SLATE_700 = colors.HexColor("#334155")      # Body text
COLOR_SLATE_500 = colors.HexColor("#64748b")      # Muted labels / borders
COLOR_SLATE_200 = colors.HexColor("#e2e8f0")      # Divider lines
COLOR_SLATE_100 = colors.HexColor("#f1f5f9")      # Table header bg
COLOR_SLATE_50 = colors.HexColor("#f8fafc")       # Card surface bg


# ---------------------------------------------------------------------------
# Helpers for Text Escaping and Verdict Colors
# ---------------------------------------------------------------------------
def safe_text(text: Optional[str]) -> str:
    """Escapes XML entities so arbitrary LLM text doesn't break ReportLab Paragraph markup."""
    if text is None:
        return ""
    escaped = html.escape(str(text))
    return escaped.replace("\n", "<br/>")


def get_verdict_colors(verdict: str):
    """Returns (text_color, bg_color, border_color) for a given verdict label."""
    v_clean = (verdict or "").strip().lower()
    if "pass" in v_clean:
        return COLOR_PASS_TEXT, COLOR_PASS_LIGHT, COLOR_PASS_BORDER
    elif "needs" in v_clean:
        return COLOR_NEEDS_TEXT, COLOR_NEEDS_LIGHT, COLOR_NEEDS_BORDER
    elif "fail" in v_clean:
        return COLOR_FAIL_TEXT, COLOR_FAIL_LIGHT, COLOR_FAIL_BORDER
    else:
        return COLOR_SLATE_700, COLOR_SLATE_100, COLOR_SLATE_200


def get_priority_colors(priority: str):
    """Returns (text_color, bg_color, border_color) for recommendation priority."""
    p_clean = (priority or "").strip().lower()
    if p_clean == "high":
        return COLOR_FAIL_TEXT, COLOR_FAIL_LIGHT, COLOR_FAIL_BORDER
    elif p_clean == "medium":
        return COLOR_NEEDS_TEXT, COLOR_NEEDS_LIGHT, COLOR_NEEDS_BORDER
    else:
        return COLOR_PRIMARY_TEXT, COLOR_PRIMARY_LIGHT, COLOR_PRIMARY_BORDER


# ---------------------------------------------------------------------------
# Numbered Canvas for Running Header and "Page X of Y" Footer
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that records pages and draws dynamic headers and 'Page X of Y' footers
    with batch ID and timestamp.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, total_pages: int):
        self.saveState()
        page_w, page_h = self._pagesize
        left_m = 36
        right_m = page_w - 36

        batch_id = getattr(self, "report_batch_id", "Batch Report")
        timestamp = getattr(self, "report_timestamp", "")

        # Running Header (Pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(COLOR_SLATE_700)
            self.drawString(left_m, page_h - 26, "LLM Evaluation Report")
            self.setFont("Helvetica", 8)
            self.setFillColor(COLOR_SLATE_500)
            self.drawString(left_m + 95, page_h - 26, f"•  Batch ID: {batch_id}")

            if timestamp:
                self.drawRightString(right_m, page_h - 26, f"Generated: {timestamp[:16].replace('T', ' ')}")

            # Header thin divider
            self.setStrokeColor(COLOR_SLATE_200)
            self.setLineWidth(0.75)
            self.line(left_m, page_h - 32, right_m, page_h - 32)

        # Running Footer (All pages)
        self.setStrokeColor(COLOR_SLATE_200)
        self.setLineWidth(0.75)
        self.line(left_m, 32, right_m, 32)

        self.setFont("Helvetica", 7.5)
        self.setFillColor(COLOR_SLATE_500)
        self.drawString(left_m, 20, "LLM Response Evaluation Framework  •  Confidential Assessment Report")
        self.drawRightString(right_m, 20, f"Page {self._pageNumber} of {total_pages}")

        self.restoreState()


# ---------------------------------------------------------------------------
# Vector Chart Builders (ReportLab Native Vector Drawings)
# ---------------------------------------------------------------------------
def build_verdict_pie_chart(stats: OverallStats) -> Drawing:
    """Builds a clean vector pie chart showing verdict distribution."""
    draw_w, draw_h = 265, 140
    d = Drawing(draw_w, draw_h)

    # Card background box
    d.add(
        Rect(
            0, 0, draw_w, draw_h,
            fillColor=COLOR_SLATE_50,
            strokeColor=COLOR_SLATE_200,
            strokeWidth=1,
            rx=6, ry=6,
        )
    )

    # Card title
    d.add(
        String(
            12, draw_h - 18,
            "Verdict Distribution",
            fontName="Helvetica-Bold",
            fontSize=9.5,
            fillColor=COLOR_SLATE_900,
        )
    )

    pie = Pie()
    pie.x = 8
    pie.y = 12
    pie.width = 95
    pie.height = 95

    # Safe data handling (ensure at least 1 non-zero value)
    p_c = max(0, stats.pass_count)
    n_c = max(0, stats.needs_improvement_count)
    f_c = max(0, stats.fail_count)
    if p_c + n_c + f_c == 0:
        p_c = 1

    pie.data = [p_c, n_c, f_c]
    pie.slices[0].fillColor = COLOR_PASS
    pie.slices[1].fillColor = COLOR_NEEDS
    pie.slices[2].fillColor = COLOR_FAIL
    pie.slices.strokeColor = colors.white
    pie.slices.strokeWidth = 1.5
    d.add(pie)

    # Legend on the right side
    legend_items = [
        (f"Pass: {stats.pass_count} ({stats.pass_percent:.1f}%)", COLOR_PASS),
        (f"Needs Imp: {stats.needs_improvement_count} ({stats.needs_improvement_percent:.1f}%)", COLOR_NEEDS),
        (f"Fail: {stats.fail_count} ({stats.fail_percent:.1f}%)", COLOR_FAIL),
    ]
    leg_x = 118
    leg_y = 88
    for label, col in legend_items:
        d.add(Rect(leg_x, leg_y - 1, 9, 9, fillColor=col, strokeColor=None, rx=2, ry=2))
        d.add(String(leg_x + 14, leg_y, label, fontName="Helvetica", fontSize=8, fillColor=COLOR_SLATE_700))
        leg_y -= 26

    return d


def build_dimension_bar_chart(stats: OverallStats) -> Drawing:
    """Builds a crisp horizontal bar chart showing average dimension scores."""
    draw_w, draw_h = 265, 140
    d = Drawing(draw_w, draw_h)

    # Card background box
    d.add(
        Rect(
            0, 0, draw_w, draw_h,
            fillColor=COLOR_SLATE_50,
            strokeColor=COLOR_SLATE_200,
            strokeWidth=1,
            rx=6, ry=6,
        )
    )

    # Card title
    d.add(
        String(
            12, draw_h - 18,
            "Average Dimension Scores",
            fontName="Helvetica-Bold",
            fontSize=9.5,
            fillColor=COLOR_SLATE_900,
        )
    )

    dims = [
        ("Relevance", stats.average_relevance_percent, colors.HexColor("#2563eb")),
        ("Accuracy", stats.average_accuracy_percent, colors.HexColor("#0284c7")),
        ("Completeness", stats.average_completeness_percent, colors.HexColor("#7c3aed")),
        ("Groundedness", stats.average_groundedness_percent, colors.HexColor("#059669")),
        ("Composite", stats.average_weighted_percent, COLOR_SLATE_900),
    ]

    y_pos = draw_h - 40
    track_w = 115
    for name, val, bar_color in dims:
        # Dimension name
        d.add(String(12, y_pos, name, fontName="Helvetica", fontSize=7.5, fillColor=COLOR_SLATE_700))

        # Bar background track
        bar_x = 88
        bar_y = y_pos - 2
        d.add(Rect(bar_x, bar_y, track_w, 8, fillColor=COLOR_SLATE_200, strokeColor=None, rx=2, ry=2))

        # Filled bar proportional to percentage
        pct = max(0.0, min(100.0, val))
        fill_w = max(2.0, track_w * (pct / 100.0))
        d.add(Rect(bar_x, bar_y, fill_w, 8, fillColor=bar_color, strokeColor=None, rx=2, ry=2))

        # Numeric percentage label
        d.add(
            String(
                bar_x + track_w + 6,
                y_pos,
                f"{pct:.1f}%",
                fontName="Helvetica-Bold",
                fontSize=7.5,
                fillColor=COLOR_SLATE_900,
            )
        )
        y_pos -= 19

    return d


# ---------------------------------------------------------------------------
# Section 1: Cover & Summary Page Builder
# ---------------------------------------------------------------------------
def build_cover_summary_page(report: BatchReportData, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    """Constructs the complete Cover & Executive Summary Page."""
    story = []

    # 1. Header Banner
    story.append(
        Paragraph("LLM RESPONSE EVALUATION REPORT", styles["DocTitle"])
    )
    story.append(
        Paragraph("Executive Quality Assessment & Batch Diagnostic Summary", styles["DocSubtitle"])
    )
    story.append(Spacer(1, 10))

    # 2. Batch Metadata Table (2-column key-value grid)
    meta = report.metadata
    succ_rate = round(meta.successful_records / max(1, meta.total_records) * 100, 1)

    meta_data = [
        [
            Paragraph(f"<b>Batch ID:</b> {safe_text(meta.batch_id)}", styles["MetaText"]),
            Paragraph(f"<b>Source Dataset:</b> {safe_text(meta.filename or 'batch.csv')}", styles["MetaText"]),
        ],
        [
            Paragraph(f"<b>Evaluation Timestamp:</b> {safe_text(meta.timestamp[:19].replace('T', ' '))}", styles["MetaText"]),
            Paragraph(f"<b>Total Submissions:</b> {meta.total_records} records", styles["MetaText"]),
        ],
        [
            Paragraph(f"<b>Successfully Evaluated:</b> {meta.successful_records} ({succ_rate}%)", styles["MetaText"]),
            Paragraph(f"<b>Ingestion Failures:</b> {meta.failed_records}", styles["MetaText"]),
        ],
    ]

    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_SLATE_50),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_SLATE_200),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_SLATE_200),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 3. KPI Summary Tiles Table (4 Primary KPI Cards)
    stats = report.overall_stats
    hal = report.hallucination_frequency

    kpi_cards_data = [
        [
            Paragraph(
                f"<font size=7 color='#065f46'><b>PASS RATE</b></font><br/>"
                f"<font size=14 color='#047857'><b>{stats.pass_percent:.1f}%</b></font><br/>"
                f"<font size=7.5 color='#334155'>{stats.pass_count} / {meta.total_records} responses</font>",
                styles["KPICell"],
            ),
            Paragraph(
                f"<font size=7 color='#92400e'><b>NEEDS IMPROVEMENT</b></font><br/>"
                f"<font size=14 color='#b45309'><b>{stats.needs_improvement_percent:.1f}%</b></font><br/>"
                f"<font size=7.5 color='#334155'>{stats.needs_improvement_count} / {meta.total_records} responses</font>",
                styles["KPICell"],
            ),
            Paragraph(
                f"<font size=7 color='#991b1b'><b>FAIL RATE</b></font><br/>"
                f"<font size=14 color='#b91c1c'><b>{stats.fail_percent:.1f}%</b></font><br/>"
                f"<font size=7.5 color='#334155'>{stats.fail_count} / {meta.total_records} responses</font>",
                styles["KPICell"],
            ),
            Paragraph(
                f"<font size=7 color='#1e40af'><b>COMPOSITE SCORE</b></font><br/>"
                f"<font size=14 color='#1d4ed8'><b>{stats.average_weighted_percent:.1f}%</b></font><br/>"
                f"<font size=7.5 color='#334155'>Overall quality index</font>",
                styles["KPICell"],
            ),
        ]
    ]

    kpi_table = Table(kpi_cards_data, colWidths=[135, 135, 135, 135])
    kpi_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), COLOR_PASS_LIGHT),
            ("BOX", (0, 0), (0, 0), 1, COLOR_PASS_BORDER),
            ("BACKGROUND", (1, 0), (1, 0), COLOR_NEEDS_LIGHT),
            ("BOX", (1, 0), (1, 0), 1, COLOR_NEEDS_BORDER),
            ("BACKGROUND", (2, 0), (2, 0), COLOR_FAIL_LIGHT),
            ("BOX", (2, 0), (2, 0), 1, COLOR_FAIL_BORDER),
            ("BACKGROUND", (3, 0), (3, 0), COLOR_PRIMARY_LIGHT),
            ("BOX", (3, 0), (3, 0), 1, COLOR_PRIMARY_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    # Secondary Hallucination KPI Strip
    hal_color_bg = COLOR_FAIL_LIGHT if hal.rate_percent >= 25.0 else COLOR_SLATE_50
    hal_color_border = COLOR_FAIL_BORDER if hal.rate_percent >= 25.0 else COLOR_SLATE_200
    hal_strip_data = [
        [
            Paragraph(
                f"<b>Hallucination Rate:</b> <font color='#b91c1c'><b>{hal.rate_percent:.1f}%</b></font> "
                f"({hal.flagged_responses_count} of {hal.total_evaluated_responses} responses flagged)  |  "
                f"<b>Total Unsupported Claims:</b> {hal.total_unsupported_claims} of {hal.total_claims_extracted} extracted  |  "
                f"<b>Groundedness Index:</b> {stats.average_groundedness_percent:.1f}%",
                styles["HallucinationStrip"],
            )
        ]
    ]
    hal_table = Table(hal_strip_data, colWidths=[540])
    hal_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), hal_color_bg),
            ("BOX", (0, 0), (-1, -1), 1, hal_color_border),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(hal_table)
    story.append(Spacer(1, 14))

    # 4. Charts Section (Side-by-side Verdict Pie Chart & Dimension Bar Chart)
    pie_drawing = build_verdict_pie_chart(stats)
    bar_drawing = build_dimension_bar_chart(stats)

    charts_table = Table([[pie_drawing, bar_drawing]], colWidths=[270, 270])
    charts_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(charts_table)

    # Clean Page Break after Cover / Executive Summary
    story.append(PageBreak())
    return story


# ---------------------------------------------------------------------------
# Section 2: Recommendations Page Builder
# ---------------------------------------------------------------------------
def build_recommendations_page(report: BatchReportData, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    """Constructs the Actionable Improvement Recommendations page."""
    story = []

    story.append(Paragraph("Actionable Improvement Recommendations", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "Empirical diagnostic guidance derived from automated scanning of recurring weaknesses, "
            "factual inconsistencies, hallucination frequencies, and dimensional scoring thresholds "
            f"across batch <b>{safe_text(report.metadata.batch_id)}</b>.",
            styles["SectionLead"],
        )
    )
    story.append(Spacer(1, 10))

    recs = report.recommendations
    if not recs:
        story.append(
            Paragraph("No significant evaluation weaknesses detected across evaluated thresholds.", styles["BodyText"])
        )
    else:
        for idx, r in enumerate(recs, 1):
            p_text, p_bg, p_border = get_priority_colors(r.priority)
            p_hex = p_text.hexval()
            
            # Header text inside card
            header_html = (
                f"<font size=8 color='{p_hex}'><b>[{r.priority.upper()} PRIORITY]</b></font>  "
                f"<font size=8 color='#1e40af'><b>{safe_text(r.category.upper())}</b></font>  •  "
                f"<font size=10 color='#0f172a'><b>{safe_text(r.headline)}</b></font>"
            )

            metric_pill = (
                f"<font size=8 color='#64748b'><b>Trigger:</b> {safe_text(r.metric_trigger)}</font>"
            )

            plain_callout = (
                f"<b>Recommendation:</b> {safe_text(r.plain_english)}"
            )

            advice_html = (
                f"<b>Remediation Engineering Advice:</b> {safe_text(r.remediation_advice)}"
            )

            card_rows = [
                [Paragraph(header_html, styles["RecCardHeader"])],
                [Paragraph(metric_pill, styles["RecTriggerText"])],
                [Paragraph(plain_callout, styles["RecCallout"])],
                [Paragraph(advice_html, styles["RecAdviceText"])],
            ]

            # Sample issues if available
            if r.sample_issues:
                sample_items_html = "<br/>".join(
                    f"&bull; <i>{safe_text(s[:180] + '...' if len(s) > 180 else s)}</i>"
                    for s in r.sample_issues[:3]
                )
                sample_block = f"<b>Observed Examples / Flagged Claims:</b><br/>{sample_items_html}"
                card_rows.append([Paragraph(sample_block, styles["RecSampleText"])])

            rec_table = Table(card_rows, colWidths=[540])
            rec_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), COLOR_SLATE_50),
                    ("BOX", (0, 0), (-1, -1), 1, p_border),
                    ("LINELEFT", (0, 0), (0, -1), 3.5, p_text),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ])
            )

            story.append(rec_table)
            story.append(Spacer(1, 10))

    story.append(PageBreak())
    return story


# ---------------------------------------------------------------------------
# Section 3: Per-Response Details Builder
# ---------------------------------------------------------------------------
def build_per_response_details_section(report: BatchReportData, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    """Constructs the Per-Response evaluation breakdown records."""
    story = []

    story.append(Paragraph("Per-Response Evaluation Details", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "Detailed assessment for each evaluated submission including model generation, "
            "verdict badges, per-dimension scores and reasoning, flagged hallucinated claims, "
            "and missing criteria.",
            styles["SectionLead"],
        )
    )
    story.append(Spacer(1, 12))

    for item in report.per_response_details:
        v_text, v_bg, v_border = get_verdict_colors(item.final_verdict)
        v_hex = v_text.hexval()
        score_pct = item.weighted_composite_score * 100.0

        # Response Header Bar Table
        header_table_data = [
            [
                Paragraph(f"<b>Response #{item.index}</b>", styles["ResponseHeaderTitle"]),
                Paragraph(f"<b>Composite Quality:</b> {score_pct:.1f}%", styles["ResponseHeaderScore"]),
                Paragraph(f"<b>VERDICT: {safe_text(item.final_verdict.upper())}</b>", styles["VerdictBadgeText"]),
            ]
        ]
        h_table = Table(header_table_data, colWidths=[160, 200, 180])
        h_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_SLATE_100),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_SLATE_200),
                ("BACKGROUND", (2, 0), (2, 0), v_bg),
                ("BOX", (2, 0), (2, 0), 1, v_border),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (2, 0), (2, 0), "CENTER"),
            ])
        )

        content_elements = [h_table, Spacer(1, 4)]

        # Prompt & Response Box
        prompt_html = f"<b>Prompt / Question:</b><br/>{safe_text(item.question)}"
        resp_html = f"<b>Model Response:</b><br/>{safe_text(item.response)}"
        box_rows = [
            [Paragraph(prompt_html, styles["PromptText"])],
            [Paragraph(resp_html, styles["ResponseContentText"])],
        ]

        if item.reference_answer:
            ref_html = f"<b>Ground Truth Reference:</b><br/>{safe_text(item.reference_answer)}"
            box_rows.insert(1, [Paragraph(ref_html, styles["ReferenceText"])])

        if item.source_document:
            src_html = f"<b>Source Document:</b><br/>{safe_text(item.source_document)}"
            box_rows.insert(2, [Paragraph(src_html, styles["ReferenceText"])])

        prompt_resp_table = Table(box_rows, colWidths=[540])
        prompt_resp_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_SLATE_50),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_SLATE_200),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_SLATE_200),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        content_elements.append(prompt_resp_table)
        content_elements.append(Spacer(1, 5))

        # Dimensional Scores & Reasoning Table (if evaluated successfully)
        if item.status == "success" and (item.relevance or item.accuracy or item.completeness or item.hallucination):
            dim_headers = [
                Paragraph("<b>Dimension</b>", styles["TableHeadText"]),
                Paragraph("<b>Score</b>", styles["TableHeadText"]),
                Paragraph("<b>Classification</b>", styles["TableHeadText"]),
                Paragraph("<b>Diagnostic Reasoning & Evidence</b>", styles["TableHeadText"]),
            ]
            dim_rows = [dim_headers]

            dims_to_show = [
                ("Relevance", item.relevance),
                ("Accuracy", item.accuracy),
                ("Completeness", item.completeness),
                ("Groundedness", item.hallucination),
            ]

            for d_name, d_obj in dims_to_show:
                if d_obj:
                    # Color code score text
                    s_pct = d_obj.score_percent
                    s_color = "#047857" if s_pct >= 75.0 else ("#b45309" if s_pct >= 50.0 else "#b91c1c")
                    cls_str = safe_text(d_obj.classification or "—")
                    reasoning_str = safe_text(d_obj.reasoning)

                    dim_rows.append([
                        Paragraph(f"<b>{d_name}</b>", styles["TableBodyText"]),
                        Paragraph(f"<font color='{s_color}'><b>{s_pct:.1f}%</b></font>", styles["TableBodyText"]),
                        Paragraph(f"<i>{cls_str}</i>", styles["TableBodyText"]),
                        Paragraph(reasoning_str, styles["TableBodyText"]),
                    ])

            dims_table = Table(dim_rows, colWidths=[90, 55, 95, 300])
            dims_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COLOR_SLATE_100),
                    ("BOX", (0, 0), (-1, -1), 1, COLOR_SLATE_200),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_SLATE_200),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ])
            )
            content_elements.append(dims_table)
            content_elements.append(Spacer(1, 5))

        # Flagged Hallucinations Box (if any)
        claims = item.flagged_claims_breakdown
        claims_strs = item.flagged_hallucinated_claims
        if claims or claims_strs:
            claim_bullets = []
            if claims:
                for c in claims:
                    st = (c.status or "unsupported").upper()
                    st_col = "#b91c1c" if st == "CONTRADICTED" else "#92400e"
                    claim_bullets.append(
                        f"&bull; <font color='{st_col}'><b>[{st}]</b></font> &ldquo;{safe_text(c.claim)}&rdquo; — "
                        f"<i>{safe_text(c.explanation)}</i>"
                    )
            else:
                for c_str in claims_strs:
                    claim_bullets.append(f"&bull; <font color='#b91c1c'><b>[UNSUPPORTED]</b></font> &ldquo;{safe_text(c_str)}&rdquo;")

            hal_box_html = (
                f"<b><font color='#991b1b'>Flagged Hallucinated Claims ({len(claim_bullets)}):</font></b><br/>"
                + "<br/>".join(claim_bullets)
            )
            hal_box = Table([[Paragraph(hal_box_html, styles["HallucinationBoxText"])]], colWidths=[540])
            hal_box.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), COLOR_FAIL_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 1, COLOR_FAIL_BORDER),
                    ("LINELEFT", (0, 0), (0, -1), 3, COLOR_FAIL),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ])
            )
            content_elements.append(hal_box)
            content_elements.append(Spacer(1, 5))

        # Missing Aspects Box (if any)
        if item.missing_aspects:
            aspect_bullets = [f"&bull; {safe_text(a)}" for a in item.missing_aspects]
            aspect_box_html = (
                f"<b><font color='#92400e'>Missing Aspects & Sub-Questions ({len(item.missing_aspects)}):</font></b><br/>"
                + "<br/>".join(aspect_bullets)
            )
            aspect_box = Table([[Paragraph(aspect_box_html, styles["MissingAspectBoxText"])]], colWidths=[540])
            aspect_box.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), COLOR_NEEDS_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 1, COLOR_NEEDS_BORDER),
                    ("LINELEFT", (0, 0), (0, -1), 3, COLOR_NEEDS),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ])
            )
            content_elements.append(aspect_box)
            content_elements.append(Spacer(1, 5))

        # Verdict Explanation Summary Strip
        v_expl_html = f"<b>Consolidated Verdict Explanation:</b> {safe_text(item.verdict_explanation)}"
        v_expl_table = Table([[Paragraph(v_expl_html, styles["VerdictExplText"])]], colWidths=[540])
        v_expl_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_SLATE_50),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_SLATE_200),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        content_elements.append(v_expl_table)
        content_elements.append(Spacer(1, 14))

        # Add all elements of this response to the story
        for el in content_elements:
            story.append(el)

    return story


# ---------------------------------------------------------------------------
# Stylesheet Configuration
# ---------------------------------------------------------------------------
def create_report_stylesheet() -> Dict[str, ParagraphStyle]:
    """Creates a unified, clean typography stylesheet."""
    base = getSampleStyleSheet()

    custom_styles = {
        "DocTitle": ParagraphStyle(
            "DocTitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=COLOR_SLATE_900,
            alignment=0,
        ),
        "DocSubtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=COLOR_SLATE_500,
            alignment=0,
        ),
        "MetaText": ParagraphStyle(
            "MetaText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=COLOR_SLATE_700,
        ),
        "KPICell": ParagraphStyle(
            "KPICell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            alignment=1,  # Center
        ),
        "HallucinationStrip": ParagraphStyle(
            "HallucinationStrip",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=COLOR_SLATE_700,
            alignment=1,  # Center
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=COLOR_SLATE_900,
            keepWithNext=True,
        ),
        "SectionLead": ParagraphStyle(
            "SectionLead",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=COLOR_SLATE_500,
            keepWithNext=True,
        ),
        "RecCardHeader": ParagraphStyle(
            "RecCardHeader",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=COLOR_SLATE_900,
        ),
        "RecTriggerText": ParagraphStyle(
            "RecTriggerText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=COLOR_SLATE_500,
        ),
        "RecCallout": ParagraphStyle(
            "RecCallout",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=COLOR_SLATE_900,
        ),
        "RecAdviceText": ParagraphStyle(
            "RecAdviceText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11.5,
            textColor=COLOR_SLATE_700,
        ),
        "RecSampleText": ParagraphStyle(
            "RecSampleText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=COLOR_SLATE_700,
        ),
        "ResponseHeaderTitle": ParagraphStyle(
            "ResponseHeaderTitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=COLOR_SLATE_900,
        ),
        "ResponseHeaderScore": ParagraphStyle(
            "ResponseHeaderScore",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=COLOR_SLATE_700,
            alignment=1,  # Center
        ),
        "VerdictBadgeText": ParagraphStyle(
            "VerdictBadgeText",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            alignment=1,  # Center
        ),
        "PromptText": ParagraphStyle(
            "PromptText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=COLOR_SLATE_900,
        ),
        "ResponseContentText": ParagraphStyle(
            "ResponseContentText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=COLOR_SLATE_700,
        ),
        "ReferenceText": ParagraphStyle(
            "ReferenceText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=COLOR_SLATE_500,
        ),
        "TableHeadText": ParagraphStyle(
            "TableHeadText",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=COLOR_SLATE_800,
        ),
        "TableBodyText": ParagraphStyle(
            "TableBodyText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=COLOR_SLATE_700,
        ),
        "HallucinationBoxText": ParagraphStyle(
            "HallucinationBoxText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=COLOR_FAIL_TEXT,
        ),
        "MissingAspectBoxText": ParagraphStyle(
            "MissingAspectBoxText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=COLOR_NEEDS_TEXT,
        ),
        "VerdictExplText": ParagraphStyle(
            "VerdictExplText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=COLOR_SLATE_700,
        ),
        "BodyText": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=COLOR_SLATE_700,
        ),
    }

    return custom_styles


# ---------------------------------------------------------------------------
# Main PDF Generator API
# ---------------------------------------------------------------------------
def generate_pdf_report(
    report_data: BatchReportData,
    output_destination: Optional[Union[str, Path, io.BytesIO]] = None,
) -> Union[bytes, str]:
    """
    Renders a complete, professional PDF report from BatchReportData.
    Accepts an output file path or BytesIO buffer.
    If output_destination is None, returns raw PDF bytes.
    """
    buffer = None
    target_path = None

    if output_destination is None:
        buffer = io.BytesIO()
        target = buffer
    elif isinstance(output_destination, io.BytesIO):
        target = output_destination
    else:
        target_path = str(output_destination)
        target = target_path

    # Page setup: letter size (612 x 792 pt), margins 36 pt left/right, 40 pt top/bottom
    doc = SimpleDocTemplate(
        target,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40,
    )

    styles = create_report_stylesheet()
    story = []

    # 1. Cover / Executive Summary Page
    story.extend(build_cover_summary_page(report_data, styles))

    # 2. Recommendations Page
    story.extend(build_recommendations_page(report_data, styles))

    # 3. Per-Response Details Section
    story.extend(build_per_response_details_section(report_data, styles))

    # Build PDF with dynamic NumberedCanvas
    def _canvas_factory(*args, **kwargs):
        c = NumberedCanvas(*args, **kwargs)
        c.report_batch_id = report_data.metadata.batch_id
        c.report_timestamp = report_data.metadata.timestamp
        return c

    doc.build(story, canvasmaker=_canvas_factory)

    logger.info(f"Successfully generated PDF report for batch {report_data.metadata.batch_id}")

    if buffer is not None:
        return buffer.getvalue()
    elif target_path is not None:
        return target_path
    else:
        return output_destination.getvalue()
