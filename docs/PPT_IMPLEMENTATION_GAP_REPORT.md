# PPT Implementation Gap Report: Skincare Routine Ingredient Checker

**Project**: Skincare Routine Ingredient Checker — An ML-Powered Ingredient Analysis & Automated MLOps System  
**Date**: 2026-10-21  
**Audit Purpose**: Identify discrepancies between the existing prototype implementation and the target Presentation (PPT) requirements, defining the concrete roadmap for corrections and extensions.

---

## 1. Executive Summary & Audit Findings

The initial project setup successfully established the MLOps skeleton:
- A verified Sephora dataset (1,472 records) with SHA-256 provenance tracking.
- An MLflow tracking setup and local SQLite registry (`mlflow.db`).
- A scikit-learn model pipeline classifying cosmetic products into 6 categories (Cleanser, Moisturizer, Face Mask, Treatment, Eye cream, Sun protect).
- A FastAPI backend and a Streamlit dashboard.
- Automated tests and Docker Compose containerization.

However, the **primary focus of the PPT** is a **Skincare Routine Ingredient Checker** and **Multi-Product Conflict Detection System** driven by **Individual Ingredient-Function Classification (NLP)**, **Product Name Fuzzy Search Lookup**, and **Packaging Label OCR ingestion**.

---

## 2. Requirement-by-Requirement Gap Matrix

| # | PPT Requirement | Existing Implementation | Gap / Missing Functionality | Impacted Files | Verification Test Needed |
|---|---|---|---|---|---|
| **1** | **Product-Name Lookup & Search Engine** | Catalog is only filtered statically in backend by skin type. No user search by name or brand. | Missing name search, fuzzy matching, brand filters, multiple match resolution, and local/external provenance. | `src/data/product_lookup.py`, `backend/app/api/routes.py`, `frontend/app.py` | `test_product_lookup_exact`, `test_product_lookup_fuzzy`, `test_product_not_found` |
| **2** | **Multi-Product Conflict Checking** | Evaluates a single synthesized routine object with simple rules. | Does not allow dynamic multi-product comparison (e.g. Cleanser A + Serum B + Cream C), cross-product active deduplication, and structured outcome categories (`NO_CONFIGURED_CONFLICT_FOUND`, `POTENTIAL_CONFLICT`, `DUPLICATE_ACTIVE`, `USER_AVOID_LIST_MATCH`, `INSUFFICIENT_INFORMATION`). | `src/rules/rule_engine.py`, `backend/app/schemas/schemas.py`, `backend/app/api/routes.py` | `test_multi_product_conflict_detection`, `test_duplicate_active_detection` |
| **3** | **Individual Chemical Ingredient Classifier (NLP)** | Only a product-level category classifier (predicting 'Cleanser', 'Moisturizer', etc. for whole product) exists. | Missing individual chemical ingredient classifier predicting cosmetic function (Humectant, Emollient, Occlusive, Exfoliating agent, Antioxidant, Surfactant, etc.) using TF-IDF word/char n-grams. | `data/processed/ingredient_taxonomy.csv`, `src/models/ingredient_classifier.py`, `scripts/train_ingredient_classifier.py`, `models/` | `test_ingredient_classification`, `test_unknown_ingredient_handling` |
| **4** | **Ingredient Synonyms & Canonical INCI Mapping** | Basic lowercase string replacement only. | No structured synonym dictionary (e.g., L-Ascorbic Acid $\leftrightarrow$ Vitamin C, Aqua $\leftrightarrow$ Water, Tocopherol $\leftrightarrow$ Vitamin E). | `data/interim/ingredient_synonyms.json`, `src/data/preprocess.py` | `test_ingredient_synonym_normalization` |
| **5** | **Packaging Label Image OCR Ingestion** | No image upload or OCR capability. | Missing file upload in Streamlit, image preprocessing, Tesseract OCR invocation, user text verification/editing, and token parsing. | `backend/app/services/ocr_service.py`, `frontend/app.py`, `backend/app/api/routes.py` | `test_ocr_extraction_and_fallback` |
| **6** | **Active Concentration Transparency** | No concentration parsing. | Need honest reporting of active % when listed, explicit `Unknown / Not provided` status when omitted, and strict avoidance of hallucinated concentrations. | `src/rules/rule_engine.py`, `backend/app/schemas/schemas.py`, `frontend/app.py` | `test_missing_concentration_handling` |
| **7** | **FastAPI Extended Endpoints** | Only `/predict`, `/check-routine`, `/model-info`, `/metrics`, `/health`, `/ready`. | Missing `/products/search`, `/products/{id}`, `/ingredients/classify`, `/ingredients/parse`, and `/ocr/extract`. | `backend/app/api/routes.py`, `backend/app/schemas/schemas.py` | `test_api_new_endpoints` |
| **8** | **Interactive Streamlit Redesign** | Single form generating fixed recommendations. | Needs Multi-Product Routine Comparison workbench, live Product Search modal, Ingredient Classifier explorer, OCR packaging uploader, and MLOps lineage metrics. | `frontend/app.py` | Manual UI verification & TestClient frontend checks |
| **9** | **MLOps Data & Prediction Drift Monitoring** | Only static operational latency/count tracking. | Missing feature distribution drift checks and unseen ingredient tracking. | `backend/app/core/metrics.py`, `src/evaluation/drift_monitor.py` | `test_drift_metrics_reporting` |

---

## 3. Implementation Plan by Phase

1. **Phase 2 — Product Search & Fuzzy Lookup**: Implement `ProductLookupService` using case-insensitive token and Levenshtein distance/fuzzy matching on `cleaned_cosmetics.csv`.
2. **Phase 3 — Multi-Product Conflict Checking**: Refactor `src/rules/rule_engine.py` to evaluate arbitrary arrays of products, emitting standardized enum statuses and source citations.
3. **Phase 4 — Ingredient-Level Function Classifier**: Create a verified INCI chemical taxonomy, train TF-IDF char/word N-gram model (Logistic Regression & candidate comparison), serialize artifact to `models/ingredient_classifier.joblib`, and track metrics.
4. **Phase 5 — Image Upload & OCR**: Implement OCR pipeline using Pillow and `pytesseract` with graceful fallback when Tesseract binary is absent.
5. **Phase 6 — Synonyms & Normalization**: Build `data/interim/ingredient_synonyms.json` and canonical mapping functions in `src/data/preprocess.py`.
6. **Phase 7 — Active Concentrations & Evidence**: Update schemas to distinguish explicit vs unstated concentrations.
7. **Phase 8 — FastAPI Route Expansion**: Implement new endpoints with full Pydantic validation.
8. **Phase 9 — Streamlit UI Redesign**: Transform Streamlit dashboard to match PPT user journey with multi-product routine comparison, product search, OCR editor, and ingredient classifier.
9. **Phase 10 — Drift Monitoring & Retraining**: Integrate unseen ingredient detection and feature drift telemetry.
10. **Phase 11 & 12 — Testing, Docker & Documentation**: Run comprehensive pytest suite (all 24 required test scenarios) and update docs.
