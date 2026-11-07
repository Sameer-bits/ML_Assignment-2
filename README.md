# Dry Bean Classification — Machine Learning Assignment 2

## Problem statement
This project compares five supervised classification algorithms on the UCI Dry Bean dataset:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The target is `Class`, which contains seven dry bean varieties. The dataset provides 16 numeric input features.

## Dataset

- Source: [UCI Machine Learning Repository — Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- Original dataset: 13,611 rows and 16 numeric features, plus the target column.
- Duplicate records removed before splitting: 68.
- Cleaned dataset used in the reported experiment: 13,543 rows.
- Missing values: none reported in the original dataset.
- Target classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA.

Features:
`Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`,
`Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`,
`roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`,
`ShapeFactor3`, `ShapeFactor4`.

## Github Repo

- Github Repository Link: https://github.com/Sameer-bits/ML_Assignment-2/tree/main

## Models Used

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9947 | 0.9307 | 0.9300 | 0.9302 | 0.9023 |
| Decision Tree | 0.8963 | 0.9464 | 0.9108 | 0.9111 | 0.9108 | 0.8746 |
| KNN | 0.9155 | 0.9832 | 0.9290 | 0.9256 | 0.9270 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9670 | 0.7736 | 0.7685 | 0.7669 | 0.7143 |
| SVM | 0.9210 | 0.9945 | 0.9342 | 0.9317 | 0.9328 | 0.9045 |

## Model Performance Observations

The five machine learning models were evaluated using Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved 91.92% accuracy and the highest AUC of 0.9947. It showed strong overall performance, with an F1-score of 0.9302 and MCC of 0.9023. |
| Decision Tree | Achieved 89.63% accuracy and an F1-score of 0.9108. It provided reasonable classification performance but performed below Logistic Regression, KNN and SVM. |
| KNN | Achieved 91.55% accuracy and an AUC of 0.9832. Its F1-score of 0.9270 indicates good predictive performance, making it a competitive model. |
| Naive Bayes | Achieved 76.30% accuracy, the lowest among the five models. Its F1-score of 0.7669 and MCC of 0.7143 indicate comparatively weaker classification performance. |
| Support Vector Machine (SVM) | Achieved the highest accuracy (92.10%), precision (0.9342), recall (0.9317), F1-score (0.9328) and MCC (0.9045). It demonstrated the strongest overall classification performance. |

### Overall Winner for the Dataset

**Support Vector Machine (SVM)** is the overall winner because it achieved the best results in five out of six evaluation metrics. However, Logistic Regression achieved the highest AUC of 0.9947. Both models demonstrated strong performance on the Dry Bean dataset.


## Metrics
Accuracy, multiclass macro one-vs-rest AUC, macro Precision, macro Recall, macro F1 and multiclass MCC.

## Install and run

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

`train_models.py` generates `test_data.csv`, `train_data.csv`, `Dry_Bean_Cleaned.csv`, the five `.joblib` models and CSV reports. The Streamlit app loads saved models; it does not train models when a test CSV is uploaded.
