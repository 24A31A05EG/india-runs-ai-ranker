# Data Dictionary: `candidates_clean.parquet`

This document describes every field in the final cleaned feature table, its type, and derivation logic.

## 1. Core Profile
| Field | Type | Description |
| :--- | :--- | :--- |
| `candidate_id` | `str` | Primary Key. |
| `name_anonymized` | `str` | Candidate's anonymized name. |
| `headline_clean` | `str` | Lowercased and stripped profile headline. |
| `total_years_exp` | `float` | Stated years of experience from profile. |
| `is_duplicate_resume`| `bool` | True if resume content (summary/career) is an exact MD5 match with another ID. |

## 2. Career Metadata
| Field | Type | Description |
| :--- | :--- | :--- |
| `num_roles` | `int` | Total number of roles in career history. |
| `avg_tenure_months` | `float` | Mean duration of all roles in months. |
| `num_career_gaps` | `int` | Number of detected gaps > 2 months between roles. |
| `has_overlapping_roles`| `bool` | True if start/end dates of roles overlap by > 1 month. |
| `job_switch_rate_per_year`| `float` | Roles per year of experience. |

## 3. Skills & Assessments
| Field | Type | Description |
| :--- | :--- | :--- |
| `skills_count` | `int` | Number of skills listed by the candidate. |
| `skills_list` | `str` | Comma-separated list of normalized skill tokens (e.g., `machine-learning`, `nlp`). |
| `assessment_count` | `int` | Number of Redrob skill assessments completed. |
| `avg_assessment_score`| `float` | Mean score across all assessments (NULL if none). |

## 4. Behavioral Signals
All 23 signals are prefixed with `sig_`. If a signal uses a sentinel value (like `-1`), it is converted to `null` and a companion `has_` boolean is added.

| Field Prefix | Type | Description |
| :--- | :--- | :--- |
| `sig_profile_completeness_score` | `float` | Level of profile completion (0-100). |
| `sig_github_activity_score` | `float` | GitHub contribution metric (NULL if no link). |
| `has_github_activity_score` | `bool` | True if a valid score exists. |
| `sig_offer_acceptance_rate` | `float` | Percentage of past offers accepted (NULL if none). |
| `has_offer_acceptance_rate` | `bool` | True if history exists. |
| ... | ... | (Includes Response Rate, Notice Period, Relocation, etc.) |
