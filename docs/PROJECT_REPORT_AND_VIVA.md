# Academic Project Report & Viva Voce Guide
**Project**: AI-Powered Skincare Routine Checker  
**Domain**: Machine Learning Operations (MLOps) & Applied NLP  

---

## 1. MLOps Lifecycle Implementation Overview

### Phase 1: Problem Definition & Feasibility
- **Objective**: Assist individuals in assessing routine composition and predicting product routine category purely based on chemical formulation and skin tolerance, augmented by evidence-backed contraindication detection.
- **Formulation**: Multi-class classification ($K=6$ categories) combined with deterministic expert rule filters.

### Phase 2: Data Acquisition & Provenance
- Ingested real datasets containing 1,472 commercial items from Sephora catalogs and Open Beauty Facts API.
- Implemented cryptographic hashing (SHA-256) and manifest tracking (`data/raw/manifest.json`) to guarantee reproducibility.

### Phase 3: Feature Engineering & Preprocessing
- Text normalization: Lowercasing, punctuation stripping, token segmentation.
- Hybrid representations: `FeatureUnion` combining TF-IDF N-grams (1-2) with scaled tabular indicators (Price, Rank, Suitability flags).

### Phase 4: Model Exploration & Experiment Tracking
- Trained interpretable Logistic Regression baseline alongside an ensemble Random Forest classifier.
- Tracked hyperparameters, metrics, and runs locally using MLflow and structured JSON telemetry.
- Winning model: `RandomForest_Classifier` achieving **54.67% accuracy** and **0.5536 Macro-F1** on held-out test splits.

### Phase 5: Gating & CI/CD Model Promotion
- Enforced strict criteria: Accuracy $\ge$ 0.40, Macro-F1 $\ge$ 0.40, and real inference sanity tests.
- Versioned serialized model artifact `models/production_model.joblib` and logged lineage in `models/model_registry.json`.

### Phase 6: Microservice Architecture & Serving
- Containerized FastAPI service exposing `/health`, `/ready`, `/model-info`, `/predict`, `/check-routine`, and `/metrics`.
- Assembled Streamlit interactive dashboard with morning/evening visualizers and downloadable Markdown reports.

---

## 2. Frequently Asked Viva / Interview Questions & Answers

### Q1: Why did you choose Random Forest over Deep Learning (e.g. BERT or Transformers)?
**Answer**: In commercial skincare ingredient lists, words are chemical names listed in descending concentration order rather than standard conversational prose with syntactic grammar. TF-IDF paired with Random Forest is computationally lightweight, operates with low latency on standard CPU environments, does not require costly GPU hardware, and provides strong multi-class classification performance without overfitting on a 1,400-sample dataset.

### Q2: How did you prevent data leakage during preprocessing?
**Answer**: We split the dataset into train, validation, and test subsets **prior** to fitting feature transformers. The `TfidfVectorizer` and `StandardScaler` are fitted exclusively on the training split inside a scikit-learn `Pipeline`, preventing test statistics or vocabulary from leaking into the training phase.

### Q3: Why separate the ML model from the rule engine?
**Answer**: Machine learning models are probabilistic and can make unpredictable mistakes on rare combinations. Clinical contraindications (e.g., combining potent retinoids with strong direct exfoliating acids or using photosensitizing ingredients without sun protection) require deterministic, evidence-reviewed guarantees. The rule engine enforces strict safety boundaries that ML predictions cannot override.

### Q4: How is model rollback handled in your MLOps pipeline?
**Answer**: The `model_registry.json` maintains historical version records and archived artifacts (`model_v1.joblib`, etc.). Executing `python scripts/retrain.py --rollback` swaps the canonical `production_model.joblib` to the latest approved prior version and records the rollback timestamp in the registry.

### Q5: What operational metrics are monitored in production?
**Answer**: We track request count, endpoint distribution, average and P95 response latency, HTTP error rate, per-category prediction frequency, and schema violation counts without storing any Personally Identifiable Information (PII).
