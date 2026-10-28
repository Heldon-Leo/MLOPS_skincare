# AI-Powered Skincare Routine Ingredient Checker — End-to-End MLOps System

An academic, production-grade Machine Learning Operations (MLOps) project that:
- **Classifies individual chemical ingredients** by cosmetic function (Humectant, Emollient, Exfoliant, Antioxidant, etc.)
- **Searches products by name** using exact, partial, and fuzzy Levenshtein matching with brand filtering
- **Detects cross-product active conflicts** (retinoid + direct AHA/BHA, benzoyl peroxide + retinoid, duplicate actives)
- **Parses packaging label OCR** from uploaded photos with Tesseract
- **Serves trained ML models** via FastAPI with full MLflow tracking and model versioning
- **Provides a Streamlit dashboard** with multi-product routine workbench and MLOps lineage view

---

## 🗂️ Project Structure

```
MLOPS_SKINCARE/
├── .github/
│   └── workflows/
│       └── ci.yml               ← Automated CI workflow running lint, validation & tests
├── data/
│   ├── raw/                     ← Real cosmetics dataset (manifest.json, cosmetics.csv)
│   ├── interim/                 ← Cleaned catalog (cleaned_cosmetics.csv), synonym dict (ingredient_synonyms.json)
│   ├── processed/               ← Train/Val/Test splits + ingredient_taxonomy.csv
│   └── README.md                ← Dataset provenance & license documentation
├── src/
│   ├── data/
│   │   ├── preprocess.py        ← Text normalization, synonym mapping, concentration extraction
│   │   ├── product_lookup.py    ← Exact/partial/fuzzy product search using rapidfuzz
│   │   └── ingredient_taxonomy.py ← 77-record INCI function labeled dataset generator
│   ├── features/
│   │   └── features.py          ← FeatureUnion (TF-IDF N-grams + tabular features)
│   ├── models/
│   │   ├── tracking.py          ← Dual MLflow + local JSON experiment tracker
│   │   └── ingredient_classifier.py ← Char N-gram TF-IDF Logistic Regression NLP model
│   ├── evaluation/
│   │   ├── evaluator.py         ← Per-class metrics + confusion matrix
│   │   └── drift_monitor.py     ← Unseen ingredient frequency & drift telemetry
│   └── rules/
│       ├── rules_data.py        ← Evidence-reviewed rule configurations
│       └── rule_engine.py       ← Multi-product conflict engine with standardized statuses
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI app with CORS middleware
│   │   ├── api/routes.py        ← All endpoints: search, classify, parse, OCR, check-routine, predict
│   │   ├── schemas/schemas.py   ← Pydantic v2 validated request/response schemas
│   │   ├── services/
│   │   │   ├── model_service.py ← Singleton production model inference service
│   │   │   └── ocr_service.py   ← Label image OCR + configurable tesseract fallback
│   │   └── core/metrics.py      ← Operational telemetry (latency, errors, prediction counts)
│   ├── Dockerfile               ← Multi-stage non-root container with tesseract-ocr
│   └── requirements.txt
├── frontend/
│   ├── app.py                   ← Streamlit dashboard (5 navigation sections)
│   ├── Dockerfile               ← Hardened non-root container with health check
│   └── requirements.txt
├── models/
│   ├── production_model.joblib  ← Active product-category classifier (RandomForest)
│   ├── model_v1.joblib          ← Versioned model artifact
│   ├── ingredient_classifier.joblib ← Individual ingredient function classifier (LR)
│   └── model_registry.json      ← Promotion audit log with rollback history
├── reports/
│   ├── model_metrics.json       ← Held-out test evaluation metrics
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── training_summary.md
│   └── ingredient_classifier_metrics.json
├── scripts/
│   ├── download_data.py         ← Dataset download with SHA-256 checksum
│   ├── validate_data.py         ← Schema & bounds validation (supports raw & interim datasets)
│   ├── train.py                 ← Multi-model training + evaluation pipeline
│   ├── promote_model.py         ← Gated model promotion (Acc ≥ 0.40, Macro-F1 ≥ 0.40)
│   └── retrain.py               ← Automated retraining + rollback workflow
├── tests/
│   ├── test_pipeline.py         ← 24 scenario tests covering all functional requirements
│   └── test_new_features.py     ← Extended tests covering brand filtering, thresholds & drift
├── docker-compose.yml           ← Dependable local orchestrator with healthchecks & non-root containers
├── .env.example                 ← Safe configuration template
├── .gitignore
├── .dockerignore                ← Lean image build exclusions
├── requirements-dev.txt         ← Local development & test requirements (includes httpx)
└── README.md
```

---

## 🚀 Quickstart — Local Native Python (Windows PowerShell)

### Prerequisites
- Python 3.11 (Python 3.11–3.13 supported)
- [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki) *(optional, for label packaging OCR; OCR gracefully falls back to manual entry when absent)*

### Step 1 — Setup Virtual Environment & Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### Step 2 — Dataset Validation (Offline-Ready)
```powershell
# Validates schema, null bounds, and data types
python scripts/validate_data.py
```
*(Note: To download fresh raw data from public upstream mirrors, run `python scripts/download_data.py`.)*

### Step 3 — Start API Backend
```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 4 — Start Streamlit Frontend
```powershell
streamlit run frontend/app.py --server.port 8501
```

### Access Points
| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Swagger Docs | http://localhost:8000/docs |
| API Health Check | http://localhost:8000/health |
| API Readiness Check | http://localhost:8000/ready |
| Model Info | http://localhost:8000/model-info |
| Operational & Drift Telemetry | http://localhost:8000/metrics |

---

## 🐳 Docker Compose (One-Command)

The repository provides a complete, isolated multi-container setup via Docker Compose.

### Start the Full Stack
```powershell
docker compose up --build
```

### Background Daemon Mode
```powershell
docker compose up -d --build
```

### Service Endpoints & Health Check
- **Streamlit Dashboard**: `http://localhost:8501` (waiting until backend is healthy before servicing traffic)
- **FastAPI API**: `http://localhost:8000` (docs at `http://localhost:8000/docs`, health at `http://localhost:8000/ready`)

### View Logs
```powershell
# View combined logs
docker compose logs -f

# View backend logs only
docker compose logs -f backend

# View frontend logs only
docker compose logs -f frontend
```

### Stop Containers
```powershell
docker compose down
```

### Rebuild from Scratch
```powershell
docker compose down -v
docker compose build --no-cache
docker compose up
```

### Docker Troubleshooting
1. **Port Conflicts (`8000` or `8501` already in use)**:
   Change port bindings in `docker-compose.yml` (e.g. `"8080:8000"` or `"8502:8501"`).
2. **Backend Unhealthy / Readiness Timeout**:
   Inspect backend startup logs via `docker compose logs backend`. The readiness check verifies that serialized ML model artifacts (`production_model.joblib`) are cleanly loaded into memory.
3. **Local Compose vs. Public Deployment**:
   This Compose setup is intended for reproducible local development and evaluation. It binds directly to host ports without SSL/TLS termination, rate limiting, or reverse-proxy authentication.

---

## 🧪 Automated Tests & CI Quality Gate

### Run All Tests Locally
```powershell
pytest tests/ -v
```

**All 28 tests pass across pipeline and regression test suites:**
- 24 end-to-end scenario tests in `tests/test_pipeline.py` (lookup, fuzzy match, missing fields, token parsing, synonym mapping, function classification, active duplication, rule warnings, avoid-list enforcement, OCR fallback, validation, rollback, and container healthchecks)
- 4 extended regression tests in `tests/test_new_features.py` (brand filtering, threshold tuning, concentration sources, drift telemetry schema)

### GitHub Actions CI
The repository includes `.github/workflows/ci.yml` which triggers automatically on pull requests and pushes to validate dataset schemas, install dependencies on Ubuntu with Tesseract, and execute the full pytest test suite.

---

## 📊 Model Evaluation & Governance

### Product Category Classifier (RandomForest)
| Metric | Value |
|---|---|
| Held-Out Accuracy | 54.67% |
| Macro F1-Score | 0.5536 |
| Weighted F1-Score | 0.5488 |

*Comparison: ~16.6% balanced random baseline across 6 product classes.*

### Ingredient Function Classifier (Logistic Regression, Char-Ngram TF-IDF)
| Metric | Value |
|---|---|
| Supported Classes | 8 (Humectant, Emollient, Occlusive, Exfoliating agent, Antioxidant, Surfactant, Soothing, Sunscreen active) |
| Training Samples | 77 (verified INCI taxonomy) |
| Validation Accuracy | 0.45 |
| Validation Macro-F1 | 0.34 |

> **Transparency & Uncertainty Notice:** The ingredient classifier is trained on a curated 77-record taxonomy. Unknown actives or low-confidence inferences are explicitly reported as `Unknown / Insufficient evidence` with full confidence scores and probability distributions rather than forcing a speculative guess.

---

## 🔗 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service liveness check |
| GET | `/ready` | Model loaded and serving |
| GET | `/model-info` | Active model version & evaluation metrics |
| GET | `/metrics` | Operational telemetry combined with drift report |
| GET | `/drift` | Detailed feature, prediction & vocabulary drift |
| GET | `/products/search?q=...` | Exact, partial, and fuzzy product name search |
| GET | `/products/{product_id}` | Product by ID lookup |
| POST | `/ingredients/parse` | Parse raw ingredient list into tokens with concentration metadata |
| POST | `/ingredients/classify` | Predict function for single ingredient with uncertainty handling |
| POST | `/ocr/extract` | OCR extract text from uploaded image (with fallback) |
| POST | `/check-routine` | Multi-product conflict & compatibility analysis |
| POST | `/predict` | Secondary product-category classifier |

---

## ⚙️ Retraining & Rollback

```powershell
# Retrain on updated dataset with automatic promotion gate (Acc >= 0.40, Macro-F1 >= 0.40)
python scripts/retrain.py

# Roll back to previous approved model version
python scripts/retrain.py --rollback
```

---

## ⚠️ Academic Disclaimer

This application is developed strictly for educational and academic demonstration. Ingredient analysis and compatibility checks do not constitute medical diagnoses, clinical prescriptions, or allergy certifications. Always patch-test new products and consult a licensed dermatologist.
