# Model Training & Experiment Summary

- **Execution Timestamp**: 2026-10-21T09:24:30.822036Z
- **Selected Architecture**: `RandomForest_Classifier`
- **Held-Out Test Accuracy**: `0.5467`
- **Held-Out Test Macro-F1**: `0.5536`
- **Held-Out Test Weighted-F1**: `0.5488`

## Candidate Model Comparison (Validation Set)

| Model Candidate | Val Accuracy | Val Macro-F1 |
|---|---|---|
| LogisticRegression_Baseline | 0.4483 | 0.4516 |
| RandomForest_Classifier | 0.4897 | 0.4652 |

## Per-Class Test Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Cleanser | 0.6600 | 0.6226 | 0.6408 | 53 |
| Eye cream | 0.6452 | 0.4878 | 0.5556 | 41 |
| Face Mask | 0.5370 | 0.5472 | 0.5421 | 53 |
| Moisturizer | 0.4217 | 0.6034 | 0.4965 | 58 |
| Sun protect | 0.6667 | 0.5882 | 0.6250 | 34 |
| Treatment | 0.5122 | 0.4200 | 0.4615 | 50 |

## Class Imbalance & Uncertainty Evaluation
- Maximum to minimum support class ratio: `1.71`
- Highest representation: `Moisturizer`
- Lowest representation: `Sun protect`
