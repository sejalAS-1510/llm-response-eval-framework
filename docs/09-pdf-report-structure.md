# PDF Report Structure & Generation Engine

## Overview

The reporting subsystem generates publication-grade PDF evaluation reports using Python's **ReportLab** library (`SimpleDocTemplate` and Platypus flowable architecture). The generator transforms aggregated `BatchReportData` objects into an auditable document suitable for executive reviews, compliance audits, and engineering post-mortems.

---

## 1. Document Architecture & Pagination

```
┌────────────────────────────────────────────────────────────────────────┐
│ Running Header: "LLM Response Evaluation Report | Batch: {batch_id}"   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  PAGE 1: COVER & EXECUTIVE SUMMARY                                     │
│  - Executive Metadata Grid (Batch ID, File, Date, Row Counts)          │
│  - KPI Summary Tiles (Pass Rate, Hallucination Rate, Weighted Score)   │
│  - Verdict Distribution Visual Table & Breakdown                       │
│  - Average Dimension Score Horizontal Bar Visualization                │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│  PAGE 2: ACTIONABLE RECOMMENDATIONS                                    │
│  - Dynamic Rule-Based Recommendations (2 to 5 items)                   │
│  - Priority Badges (HIGH / MEDIUM / LOW)                               │
│  - Empirical Metric Triggers (Exact % and count affected)              │
│  - Targeted Remediation Engineering Actions                            │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│  PAGE 3+: PER-RESPONSE DETAILED EVALUATION CARDS                       │
│  - Record Header Card with Color-Coded Verdict Badge                   │
│  - Prompt, Model Generation & Reference Ground Truth                   │
│  - Dimensional Breakdown Table (Score, Classification, Reasoning)      │
│  - Flagged Hallucination Warnings & Unsupported Claims                 │
│  - Missing Aspects & Sub-question Omissions                            │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│ Running Footer: "Confidential | Page X of Y"                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Two-Pass `NumberedCanvas` Pagination

Standard ReportLab canvases cannot determine the total page count upfront. To render exact `"Page X of Y"` footers and running headers across hundreds of pages, the engine overrides `canvas.Canvas` with a two-pass accumulator:

```python
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        # Save page state in memory for second pass
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()
```

### Canvas Elements Rendered
- **Running Header**: Top banner showing document title, batch ID, and evaluation date.
- **Running Footer**: Bottom divider line with framework branding and dynamic `"Page X of Y"` right-aligned page counter.

---

## 3. Structural Sections

### Section 1: Executive Cover & KPI Summary
1. **Title Banner**: Framework branding, evaluation timestamp, and source CSV filename.
2. **Metadata Table**: Total rows processed, successful evaluations, and failed/skipped rows.
3. **Executive KPI Cards**: Prominent callout cards displaying:
   - **Pass Rate %**: Color-coded emerald badge.
   - **Hallucination Rate %**: Flagged percentage and claim count.
   - **Mean Composite Score**: Overall weighted performance.
4. **Verdict Distribution**: Tabular and graphical representation of Pass, Needs Improvement, and Fail allocations.
5. **Dimension Scores**: Comparative visual breakdown across Relevance, Accuracy, Completeness, and Groundedness.

### Section 2: Dynamic Recommendations Engine ([`src/reporting/recommendations.py`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/src/reporting/recommendations.py))
The recommendations module scans the aggregated batch metrics against empirical frequency thresholds to generate 2 to 5 actionable improvements:

| Category | Priority | Trigger Condition | Example Remediation |
| :--- | :---: | :--- | :--- |
| **Grounding** | `HIGH` | Hallucination rate $> 15\%$ | Implement strict retrieval grounding prompts and enforce citations. |
| **Numeric Claims** | `HIGH` | $> 10\%$ unsupported numbers | Add pre-generation calculator tools or strict entity extractors for statistics. |
| **Completeness** | `MEDIUM` | Incompleteness $> 15\%$ | Inject system prompts instructing models to address every sub-question. |
| **Accuracy** | `HIGH` | Inaccuracy $> 15\%$ | Fine-tune embedding retrieval or verify reference dataset quality. |
| **Relevance** | `MEDIUM` | Off-topic rate $> 15\%$ | Refactor prompt templates to penalize generic conversational deflection. |

### Section 3: Per-Response Evaluation Cards
Every evaluated record is rendered as an isolated, styled evaluation card:
1. **Card Header**: Row index, evaluation status, and a colored verdict pill (`Pass`, `Needs Improvement`, `Fail`).
2. **Context Blocks**: The user prompt, AI model response, and reference answer formatted inside background-shaded callout boxes.
3. **Dimensional Breakdown Table**:
   - Scores expressed both as decimals ($0.0–1.0$) and percentages ($0–100\%$).
   - Categorical classifications (`fully_relevant`, `correct`, `partially_complete`, etc.).
   - Full diagnostic reasoning from each judge agent.
   - Quoted citations or evidence extracts from the reference text.
4. **Issue Callouts**:
   - **Flagged Hallucinations**: Amber/Red callout boxes listing discrete ungrounded claims.
   - **Missing Aspects**: Bulleted tags enumerating omitted sub-questions.

---

## 4. Typography & Defensive Text Layout

To guarantee that long reasoning text flows naturally across page boundaries without overlapping or clipping:

### A. Paragraph Leading & Font Sizing
ReportLab requires explicit `leading` (line height) proportional to `fontSize`. Failing to specify `leading` causes multi-line paragraphs to collide. All styles maintain a ratio of at least $1.25\times$:
```python
styles.add(ParagraphStyle(
    name="CardReasoning",
    parent=styles["Normal"],
    fontSize=8.5,
    leading=11.5,
    textColor=COLOR_SLATE_700,
))
```

### B. XML Entity Escaping
Arbitrary text generated by LLMs frequently contains unescaped HTML characters (`<`, `>`, `&`). The helper `safe_text()` escapes these entities before instantiating `Paragraph` flowables:
```python
def safe_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    escaped = html.escape(str(text))
    return escaped.replace("\n", "<br/>")
```

### C. Orphan Prevention (`KeepTogether`)
Card headers, metadata rows, and score summary tables are wrapped in `KeepTogether` blocks to prevent awkward page splits where a title appears at the bottom of a page and its table appears on the next.

---

## 5. Design Tokens & Color Palette

The PDF report adheres strictly to the design system used across the web dashboard:

| Token Name | Hex Code | Visual Application |
| :--- | :--- | :--- |
| `COLOR_PRIMARY` | `#1d4ed8` | Document title, table headers, primary accents |
| `COLOR_PASS` | `#047857` | Pass verdict pills, high score indicators |
| `COLOR_NEEDS` | `#b45309` | Needs Improvement badges, medium priority warnings |
| `COLOR_FAIL` | `#b91c1c` | Fail verdict pills, severe hallucination alerts |
| `COLOR_SLATE_900` | `#0f172a` | Section titles and primary headers |
| `COLOR_SLATE_700` | `#334155` | Primary body and diagnostic reasoning text |
| `COLOR_SLATE_500` | `#64748b` | Muted labels, row indices, timestamps |
| `COLOR_SLATE_50` | `#f8fafc` | Background fill for response cards and quote blocks |
