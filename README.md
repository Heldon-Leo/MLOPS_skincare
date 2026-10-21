# AI-Powered Skincare Routine Ingredient Checker — End-to-End MLOps System

An academic, production-grade Machine Learning Operations (MLOps) project that:
- **Classifies individual chemical ingredients** by cosmetic function (Humectant, Emollient, Exfoliant, Antioxidant, etc.)
- **Searches products by name** using exact, partial, and fuzzy Levenshtein matching
- **Detects cross-product active conflicts** (retinoid + direct AHA/BHA, benzoyl peroxide + retinoid, duplicate actives)
- **Parses packaging label OCR** from uploaded photos with Tesseract
- **Serves trained ML models** via FastAPI with full MLflow tracking and model versioning
- **Provides a Streamlit dashboard** with multi-product routine workbench and MLOps lineage view

---

## 🗂️ Project Structure

```
d:\MLOPS_SKINCARE\
├── data/
│   ├── raw/            ← Real cosmetics dataset (cosmetics.csv, OBF API sample)
│   ├── interim/        ← Cleaned catalog (cleaned_cosmetics.csv), synonym dict (ingredient_synonyms.json)
│   ├── processed/      ← Train/Val/Test splits + ingredient_taxonomy.csv
│   └── README.md       ← Dataset provenance & license documentation
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
│   │   │   ├── model_service.py  ← Singleton production model inference service
│   │   │   └── ocr_service.py    ← Label image OCR + graceful tesseract fallback
│   │   └── core/metrics.py      ← Operational telemetry (latency, errors, prediction counts)
│   ├── tests/
│   ├── Dockerfile               ← Includes tesseract-ocr system package
│   └── requirements.txt
├── frontend/
│   ├── app.py                   ← Streamlit dashboard (5 navigation sections)
│   ├── Dockerfile
│   └── requirements.txt
├── models/
│   ├── production_model.joblib  ← Active product-category classifier (RandomForest)
│   ├── model_v1.joblib          ← Versioned model artifact
│   ├── ingredient_classifier.joblib ← Individual ingredient function classifier (LR)
│   └── model_registry.json     ← Promotion audit log with rollback history
├── reports/
│   ├── model_metrics.json       ← Held-out test evaluation metrics
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── training_summary.md
│   └── ingredient_classifier_metrics.json
├── scripts/
│   ├── download_data.py    ← Dataset download with SHA-256 checksum
│   ├── validate_data.py    ← Schema & bounds validation
│   ├── train.py            ← Multi-model training + evaluation pipeline
│   ├── promote_model.py    ← Gated model promotion (Acc ≥ 0.40, Macro-F1 ≥ 0.40)
│   └── retrain.py          ← Automated retraining + rollback workflow
├── tests/
│   └── test_pipeline.py    ← 24 pytest tests covering all PPT requirements
├── docs/
│   ├── PPT_IMPLEMENTATION_GAP_REPORT.md
│   └── PROJECT_REPORT_AND_VIVA.md
├── data/interim/
│   └── ingredient_synonyms.json  ← Versioned cosmetic synonym mappings
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .dockerignore
├── requirements-dev.txt
└── README.md
```

---

## 🚀 Quickstart — Local Native Python (Windows PowerShell)

### Prerequisites
- Python 3.11
- [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki) *(optional, for label OCR)*

### Step 1 — Install Dependencies
```powershell
pip install -r requirements-dev.txt
```

### Step 2 — Download Dataset & Train Models
```powershell
# Download real cosmetics dataset from verified source
python scripts/download_data.py

# Validate schema and bounds
python scripts/validate_data.py

# Preprocess and split data
python src/data/preprocess.py

# Generate ingredient function taxonomy dataset
python src/data/ingredient_taxonomy.py

# Train product-category classifier (RandomForest & LR baseline)
python scripts/train.py

# Train individual ingredient function classifier
python src/models/ingredient_classifier.py

# Promote best model to production (gated: Acc>=0.40, Macro-F1>=0.40)
python scripts/promote_model.py
```

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
| Health Check | http://localhost:8000/health |
| Model Info | http://localhost:8000/model-info |
| Operational Metrics | http://localhost:8000/metrics |

---

## 🐳 Docker Compose (One-Command)

```powershell
docker compose up --build
```
- `skincare_api_backend` → `http://localhost:8000`
- `skincare_streamlit_frontend` → `http://localhost:8501`

> **Note:** Tesseract OCR is pre-installed in the Docker backend image (`tesseract-ocr-eng`).

---

## 🧪 Automated Tests

```powershell
pytest tests/test_pipeline.py -v
```

**All 24 tests pass (5.79s).** Coverage includes:
| # | Test Scenario |
|---|---|
| 1 | Exact product name lookup |
| 2 | Partial name lookup |
| 3 | Case-insensitive matching |
| 4 | Fuzzy typo-tolerant search |
| 5 | Ambiguous multi-match selection |
| 6 | Not-found handling |
| 7 | Missing ingredient list detection |
| 8 | Manual ingredient parsing |
| 9 | Synonym normalization (Aqua→Water, Nicotinamide→Niacinamide) |
| 10 | Ingredient function classification (Glycerin→Humectant) |
| 11 | Unknown ingredient fallback |
| 12 | Duplicate active across products |
| 13 | Configured conflict detection (Retinoid + AHA) |
| 14 | Avoid-list enforcement |
| 15 | Concentration extraction (explicit % vs Unknown) |
| 16 | OCR fallback when binary absent |
| 17 | FastAPI request validation (422 on bad payload) |
| 18 | Existing endpoint compatibility |
| 19 | Production model loading & inference |
| 20 | MLflow registry version consistency |
| 21 | Promotion gate threshold enforcement |
| 22 | Rollback function callable |
| 23 | Frontend→Backend integration via TestClient |
| 24 | Docker healthcheck endpoint |

---

## 📊 Model Evaluation Results

### Product Category Classifier (RandomForest)
| Metric | Value |
|---|---|
| Held-Out Accuracy | 54.67% |
| Macro F1-Score | 0.5536 |
| Weighted F1-Score | 0.5488 |

*Comparison: ~16.6% balanced random baseline across 6 classes*

### Ingredient Function Classifier (Logistic Regression, Char-Ngram TF-IDF)
| Metric | Value |
|---|---|
| Supported Classes | 8 (Humectant, Emollient, Occlusive, Exfoliating agent, Antioxidant, Surfactant, Soothing, Sunscreen active) |
| Training Samples | 77 (verified INCI taxonomy) |
| Validation Accuracy | 0.45 |
| Validation Macro-F1 | 0.34 |

> **Limitation Note:** The ingredient classifier is trained on a curated 77-record taxonomy. Performance increases proportionally with more labeled records. Unknown actives are correctly reported as `Unknown / Insufficient evidence`.

---

## 🔄 MLOps Workflow

```
Dataset Download (SHA-256 verified) 
  → Validation (schema + bounds) 
  → Preprocessing (synonym normalization + train/val/test split) 
  → Feature Engineering (TF-IDF + tabular FeatureUnion) 
  → Multi-Model Training (LR baseline vs RandomForest) 
  → MLflow Experiment Tracking 
  → Gated Promotion (Acc≥0.40, Macro-F1≥0.40 + inference sanity) 
  → Versioned Model Registry (model_registry.json) 
  → FastAPI Production Serving 
  → Drift Monitoring (unseen ingredient rate)
  → Rollback on Demand (scripts/retrain.py --rollback)
```

---

## 🔗 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service liveness check |
| GET | `/ready` | Model loaded and serving |
| GET | `/model-info` | Active model version & metrics |
| GET | `/metrics` | Operational telemetry |
| GET | `/products/search?q=...` | Fuzzy product name search |
| GET | `/products/{product_id}` | Product by ID |
| POST | `/ingredients/parse` | Parse raw ingredient list into tokens |
| POST | `/ingredients/classify` | Predict function for single ingredient |
| POST | `/ocr/extract` | OCR extract text from uploaded image |
| POST | `/check-routine` | Multi-product conflict & compatibility analysis |
| POST | `/predict` | Legacy product-category classifier |

---

## ⚙️ Retraining & Rollback

```powershell
# Retrain on updated dataset with automatic gating
python scripts/retrain.py

# Roll back to previous approved model version
python scripts/retrain.py --rollback
```

---

## ⚠️ Academic Disclaimer

This application is developed for educational and academic demonstration. Ingredient analysis results do not constitute medical diagnoses, clinical prescriptions, or allergy certifications. Always patch-test new products and consult a licensed dermatologist.
