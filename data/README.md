# Dataset Provenance and Documentation

## 1. Primary Dataset: Sephora Cosmetics Products & Ingredients
- **Verified Source URL**: [DataCamp Cosmetics Dataset on GitHub](https://raw.githubusercontent.com/veeralakrishna/DataCamp-Project-Solutions-Python/master/Comparing%20Cosmetics%20by%20Ingredients/datasets/cosmetics.csv) / [Kaggle Sephora Products Dataset](https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets)
- **Local Location**: `data/raw/cosmetics.csv`
- **Acquisition Date**: 2026-10-21
- **File Format**: CSV (UTF-8)
- **Size**: 1,472 records, 11 features
- **License**: CC BY-SA / MIT Open Educational Reference
- **SHA-256 Checksum**: `37589163907b20bfdb6f1e3c610b789b35b15e80cce3ef687f0f65a502618857`

### Schema Overview:
| Column | Type | Description | Sample Values |
|---|---|---|---|
| `Label` | string | Skincare category target | `Moisturizer`, `Cleanser`, `Face Mask`, `Treatment`, `Eye cream`, `Sun protect` |
| `Brand` | string | Brand name | `LA MER`, `CLINIQUE`, `DRUNK ELEPHANT` |
| `Name` | string | Commercial product name | `Crème de la Mer`, `Liquid Facial Soap` |
| `Price` | float | Retail price in USD | `$3.00` to `$370.00` (Mean: $55.58) |
| `Rank` | float | Consumer rating (0.0 to 5.0) | Mean: 4.15 |
| `Ingredients` | string | Full INCI ingredient list | Comma-separated chemical names |
| `Combination` | int (0/1) | Suitable for combination skin | 0 or 1 |
| `Dry` | int (0/1) | Suitable for dry skin | 0 or 1 |
| `Normal` | int (0/1) | Suitable for normal skin | 0 or 1 |
| `Oily` | int (0/1) | Suitable for oily skin | 0 or 1 |
| `Sensitive` | int (0/1) | Suitable for sensitive skin | 0 or 1 |

### Target Distribution:
- `Moisturizer`: 298
- `Cleanser`: 281
- `Face Mask`: 266
- `Treatment`: 248
- `Eye cream`: 209
- `Sun protect`: 170
- **Total**: 1,472

---

## 2. Secondary Source: Open Beauty Facts API
- **Verified Source URL**: `https://world.openbeautyfacts.org/api/v2/`
- **Local Sample**: `data/raw/openbeautyfacts_sample.json`
- **Acquisition Method**: Live REST API query (`/api/v2/search?categories_tags_en=face-creams`)
- **License**: Open Database License (ODbL)

---

## 3. Data Integrity & Biases
- **Biases**: Commercial prestige/retail bias towards items stocked in Sephora; pricing skews higher ($55 average).
- **Limitations**: Ingredient lists do not specify exact % concentrations (trade secret protections in cosmetics).
- **Academic MLOps Utility**: Highly suitable for real multi-class skincare category inference, ingredient profiling, compatibility verification, and rule-based routine assembly.
