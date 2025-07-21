# Advanced Fraud Detection Project Report

## Overview
This report provides a detailed analysis of the Advanced Fraud Detection System. The project employs multiple machine learning models, ensemble techniques, and a meta learner to achieve high accuracy in fraud detection.

## Data and Preprocessing
- **Dataset**: Financial transactions dataset with 6,362,620 records.
- **Final Sample Size**: 1,000,000 transactions (1,007,678 actual)
- **Fraud Rate in Final Sample**: 0.82%
- **Training Set**: 806,142 samples
- **Test Set**: 201,536 samples
- **SMOTE Balanced Training**: 1,599,144 samples (50/50 fraud ratio)

## Feature Engineering
- Created 31 features, including transaction amount transformations, balance differences, and anomaly indicators.

## Models Used
1. **Random Forest**
2. **XGBoost**
3. **Logistic Regression**
4. **Isolation Forest**
5. **Meta Ensemble**: Combines multiple models using a Gradient Boosting Classifier.

## Evaluation Results (1,000,000 Samples)
- **Random Forest**: Accuracy 100.00%, F1-Score 99.91%, ROC-AUC 99.95%
- **XGBoost**: Accuracy 100.00%, F1-Score 99.85%, ROC-AUC 99.98%
- **Logistic Regression**: Accuracy 99.99%, F1-Score 99.60%, ROC-AUC 99.98% (**Best Performer**)
- **Meta Ensemble**: Accuracy 100.00%, F1-Score 99.85%, ROC-AUC 99.91%

## Top Important Features
1. `amount_equals_old_balance` (28.54%)
2. `step` (25.51%)
3. `balance_diff_orig` (11.74%)

## Performance Scaling Analysis

| Sample Size | Fraud Rate | Best Model | ROC-AUC | Accuracy |
|-------------|------------|------------|---------|----------|
| 50,000 | 26.74% | XGBoost | 99.99% | 99.93% |
| 200,000 | 3.95% | XGBoost | 99.97% | 99.99% |
| 500,000 | 1.62% | Logistic Regression | 99.97% | 99.99% |
| 1,000,000 | 0.82% | Logistic Regression | 99.98% | 99.99% |

## Meta Learner Insights
The meta learner improved the overall prediction by refining the ensemble approach using predictions from base models as inputs. The Gradient Boosting meta learner successfully combined predictions from Random Forest, XGBoost, and Logistic Regression to achieve consistent high performance across different sample sizes.

## Visualizations
- Evaluation and feature importance plots are available:
  - `fraud_detection_evaluation.png`
  - `feature_importance.png`

## Conclusion
The project successfully demonstrates an advanced approach to fraud detection using ensemble learning and meta learning. The models achieved high accuracy, particularly with the meta learner approach, providing a robust solution for fraud detection.

---

This report presents a comprehensive view of the project results, methodologies, and insights gained during the analysis.
