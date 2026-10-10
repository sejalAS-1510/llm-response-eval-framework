# Internship Agile Documentation & Quality Assurance Artifacts

This directory contains the official Infosys Springboard Agile lifecycle and Quality Assurance (QA) documentation for the **LLM Response Evaluation Platform**.

---

## 📁 Artifact Index

| Artifact File | Description | Scope & Coverage |
| :--- | :--- | :--- |
| **[`Agile_Template_v0.1.xls`](Agile_Template_v0.1.xls)** | End-to-end Agile Scrum lifecycle workbook across 4 Sprints | • **Product Backlog**: 16 User Stories (MoSCoW prioritized, US01–US16)<br>• **Sprint Backlog**: 28 Sized engineering tasks with Monday–Friday schedule & daily burndown effort<br>• **Daily Standup**: **40 Standup logs** strictly following weekly **Monday–Friday** schedule (Days 01–10 across Sprints 1–4)<br>• **Retrospection**: 4 Sprint retrospectives (Start/Stop/Continue/Action Items) |
| **[`Defect_Tracker Template_v0.1.xlsx`](Defect_Tracker%20Template_v0.1.xlsx)** | Industry-standard defect log and resolution ledger | • **13 Documented Defects** spanning Sprints 1 to 4 with all submission and action-taken dates on Monday–Friday workdays<br>• Validated defect taxonomy: `Logical`, `Maintainability`, `Others`, `Standards`, `User Interface`<br>• Full triage metadata: Submitter, Detection date, Severity/Type, Root cause analysis, Action taken, Resolution date, and Verification status (**100% Closed**) |
| **[`Unit_Test_Plan_v0.1.xlsx`](Unit_Test_Plan_v0.1.xlsx)** | Comprehensive Unit and Integration Test Plan | • **30 Test Case Specifications** (`TC_UT_01` to `TC_UT_30`) covering all 86 automated test assertions (`pytest`)<br>• Details: Test procedure, Condition to be tested, Expected Result, and Actual Result (**100% Pass**) |

---

## 🗓️ Sprint Alignment & Workday Calendar

All four 2-week Agile Sprints strictly adhere to regular business workdays (**Monday to Friday**, 10 working days per sprint, 40 working days total):

| Sprint | Milestone Focus | Workday Date Range | Standup Days | Engineering Tasks |
| :---: | :--- | :---: | :---: | :---: |
| **Sprint 1** | **Milestone 1**: Input Validation Gateway, SQLite DB, ChromaDB & Chunking | Mon 17-Aug-2026 – Fri 28-Aug-2026 | Day 01 (17-Aug) to Day 10 (28-Aug) | T1.1 – T1.7 (7 tasks, 52 hrs) |
| **Sprint 2** | **Milestone 2**: Multi-Agent Core (Relevance, Accuracy, Hallucination) & Offline Engine | Mon 31-Aug-2026 – Fri 11-Sep-2026 | Day 01 (31-Aug) to Day 10 (11-Sep) | T2.1 – T2.7 (7 tasks, 54 hrs) |
| **Sprint 3** | **Milestone 3**: Completeness Judge, Weighted Verdict Model & CSV Batch Evaluator | Mon 14-Sep-2026 – Fri 25-Sep-2026 | Day 01 (14-Sep) to Day 10 (25-Sep) | T3.1 – T3.7 (7 tasks, 52 hrs) |
| **Sprint 4** | **Milestone 4**: SaaS Web Dashboard, Longitudinal Trends, ReportLab PDF & Deployment | Mon 28-Sep-2026 – Fri 09-Oct-2026 | Day 01 (28-Sep) to Day 10 (09-Oct) | T4.1 – T4.7 (7 tasks, 50 hrs) |

---

## 👥 Scrum Team Roles & Ceremonies

- **Scrum Master & Lead Engineer**: Sejal Shinkar
- **Sprint Length**: 2 Weeks (10 Working Days, Monday – Friday)
- **Daily Standups**: Weekly Monday – Friday (15 minutes daily) documenting progress, technical impediments, and resolutions
- **Sprint Reviews & Retrospectives**: Conducted on Day 10 (alternate Fridays) at milestone completion
- **QA Verification**: 100% automated test coverage with 86 passing tests and zero open defects
