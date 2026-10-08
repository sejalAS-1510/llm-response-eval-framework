# Dashboard Metric Calculation Methodology

## Overview

The interactive dashboard and backend analytics engines compute high-level Key Performance Indicators (KPIs) to summarize evaluation runs. To ensure mathematical integrity, all dimensional score averages and rate percentages are computed strictly over **successfully evaluated records**, insulating metrics from unparseable or failed CSV lines.

---

## 1. Volume & Execution Counts

Let $I$ be the set of all rows ingested from the uploaded CSV:

$$N_{\text{total}} = |I|$$

The dataset is partitioned into successful evaluations and ingestion/runtime failures:

$$I_{\text{succ}} = \{ i \in I \mid i.\text{status} = \text{"success"} \wedge i.\text{result} \ne \text{null} \}$$
$$I_{\text{fail}} = \{ i \in I \mid i.\text{status} \ne \text{"success"} \vee i.\text{result} = \text{null} \}$$

$$N_{\text{succ}} = |I_{\text{succ}}|, \quad N_{\text{fail}} = |I_{\text{fail}}|$$

$$N_{\text{total}} = N_{\text{succ}} + N_{\text{fail}}$$

---

## 2. Verdict Distribution Metrics

Every record $i \in I_{\text{succ}}$ receives a synthesized verdict:

$$v_i \in \{\text{"Pass"}, \text{"Needs Improvement"}, \text{"Fail"}\}$$

### Verdict Counts
$$C_{\text{pass}} = \sum_{i \in I_{\text{succ}}} \mathbf{1}(v_i = \text{"Pass"})$$
$$C_{\text{needs}} = \sum_{i \in I_{\text{succ}}} \mathbf{1}(v_i = \text{"Needs Improvement"})$$
$$C_{\text{fail}} = \sum_{i \in I_{\text{succ}}} \mathbf{1}(v_i = \text{"Fail"})$$

$$\text{Identity Check: } C_{\text{pass}} + C_{\text{needs}} + C_{\text{fail}} = N_{\text{succ}}$$

### Percentage Formulations
In the dashboard KPI cards and `BatchStatistics`, percentages evaluate the quality rate of valid model outputs:

$$\text{Pass Rate \%} = \begin{cases} 
\text{round}\left(\frac{C_{\text{pass}}}{N_{\text{succ}}} \times 100, 1\right) & \text{if } N_{\text{succ}} > 0 \\ 
0.0 & \text{if } N_{\text{succ}} = 0 
\end{cases}$$

$$\text{Needs Improvement \%} = \begin{cases} 
\text{round}\left(\frac{C_{\text{needs}}}{N_{\text{succ}}} \times 100, 1\right) & \text{if } N_{\text{succ}} > 0 \\ 
0.0 & \text{if } N_{\text{succ}} = 0 
\end{cases}$$

$$\text{Fail \%} = \begin{cases} 
\text{round}\left(\frac{C_{\text{fail}}}{N_{\text{succ}}} \times 100, 1\right) & \text{if } N_{\text{succ}} > 0 \\ 
0.0 & \text{if } N_{\text{succ}} = 0 
\end{cases}$$

---

## 3. Hallucination Frequency Metrics

Hallucination detection evaluates both response-level risk and claim-level precision.

### Response-Level Hallucination Rate
A response is flagged if at least one atomic claim is unsupported or contradicted:

$$H_{\text{flagged}} = \sum_{i \in I_{\text{succ}}} \mathbf{1}(h_i = 1)$$

*(where indicator $h_i = 1$ if response $i$ has ungrounded claims (`is_hallucinated == True`), else $0$)*

$$\text{Hallucination Rate \%} = \begin{cases} 
\text{round}\left(\frac{H_{\text{flagged}}}{N_{\text{succ}}} \times 100, 1\right) & \text{if } N_{\text{succ}} > 0 \\ 
0.0 & \text{if } N_{\text{succ}} = 0 
\end{cases}$$

### Claim-Level Granularity
For deep auditing in reports:
- **Total Claims Extracted**:
  $$M_{\text{total}} = \sum_{i \in I_{\text{succ}}} M_{\text{claims}, i}$$
  *(sum of `total_claims` across all valid responses)*
- **Total Unsupported Claims**:
  $$M_{\text{unsupported}} = \sum_{i \in I_{\text{succ}}} M_{\text{unsupported}, i}$$
  *(sum of `unsupported_claims_count` across all valid responses)*

---

## 4. Dimensional Score Averages

All continuous dimensional scores are averaged across valid responses and rounded to 3 decimal places (displayed as 0–100% on UI charts):

### Average Relevance Score
$$\overline{S}_{\text{rel}} = \frac{1}{N_{\text{succ}}} \sum_{i \in I_{\text{succ}}} S_{\text{rel}, i}$$
*(where $S_{\text{rel}, i}$ is `result.relevance.score` for item $i$)*

### Average Accuracy Score
$$\overline{S}_{\text{acc}} = \frac{1}{N_{\text{succ}}} \sum_{i \in I_{\text{succ}}} S_{\text{acc}, i}$$
*(where $S_{\text{acc}, i}$ is `result.accuracy.score` for item $i$)*

### Average Completeness Score
$$\overline{S}_{\text{comp}} = \frac{1}{N_{\text{succ}}} \sum_{i \in I_{\text{succ}}} S_{\text{comp}, i}$$
*(where $S_{\text{comp}, i}$ is `result.completeness.score` for item $i$)*

### Average Groundedness Score
Groundedness is the arithmetic complement of the hallucination score ($S_{\text{hal}, i}$):
$$G_i = \max\left(0.0, 1.0 - S_{\text{hal}, i}\right)$$

$$\overline{S}_{\text{grd}} = \frac{1}{N_{\text{succ}}} \sum_{i \in I_{\text{succ}}} G_i$$
*(where $S_{\text{hal}, i}$ is `result.hallucination.hallucination_score` for item $i$)*

### Average Weighted Composite Score
$$\overline{S}_{\text{weighted}} = \frac{1}{N_{\text{succ}}} \sum_{i \in I_{\text{succ}}} S_{\text{weighted}, i}$$
*(where $S_{\text{weighted}, i}$ is `result.verdict.weighted_score` for item $i$)*

---

## 5. Longitudinal Quality Trends (`GET /api/v1/trends`)

The framework tracks evaluation metrics across historical batches to determine model drift or improvements over successive releases.

### Quality Trajectory Classification
When multiple batches exist, the trajectory is computed by comparing the latest batch's average weighted score ($\overline{S}_{\text{curr}}$) against the preceding batch ($\overline{S}_{\text{prev}}$):

$$\Delta S_{\%} = \left(\frac{\overline{S}_{\text{curr}} - \overline{S}_{\text{prev}}}{\overline{S}_{\text{prev}}}\right) \times 100$$

$$\text{Trajectory} = \begin{cases} 
\text{"improving"} & \text{if } \Delta S_{\%} > +2.0\% \\ 
\text{"degrading"} & \text{if } \Delta S_{\%} < -2.0\% \\ 
\text{"stable"} & \text{if } -2.0\% \le \Delta S_{\%} \le +2.0\% \\ 
\text{"insufficient data"} & \text{if } \text{Total Batches} < 2 
\end{cases}$$
*(returns `"insufficient_data"` in API response if fewer than 2 batches exist)*
