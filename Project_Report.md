# Advanced Fraud Detection Project Report

## Overview
This report provides a detailed analysis of the Advanced Fraud Detection System. The project employs multiple machine learning models, ensemble techniques, and a meta learner to achieve high accuracy in fraud detection.

## Data and Preprocessing
- **Dataset**: Financial transactions dataset with 6,362,620 records.
- **Sample Size for Analysis**: 50,000
- **Fraud Rate in Sample**: 26.74%

## Feature Engineering
- Created 31 features, including transaction amount transformations, balance differences, and anomaly indicators.

## Models Used
1. **Random Forest**
2. **XGBoost**
3. **Logistic Regression**
4. **Isolation Forest**
5. **Meta Ensemble**: Combines multiple models using a Gradient Boosting Classifier.

## Evaluation Results
- **Random Forest**: Accuracy 99.97%, F1-Score 99.94%, ROC-AUC 99.97%
- **XGBoost**: Accuracy 99.93%, F1-Score 99.88%, ROC-AUC 99.99%
- **Logistic Regression**: Accuracy 99.85%, F1-Score 99.73%, ROC-AUC 99.92%
- **Meta Ensemble**: Accuracy 99.93%, F1-Score 99.88%, ROC-AUC 99.94%

## Key Features
- `step`: Significant predictor in fraud detection.

## Meta Learner Insights
The meta learner improved the overall prediction by refining the ensemble approach using predictions from base models as inputs.

## Visualizations
- Evaluation and feature importance plots are available:
  - `fraud_detection_evaluation.png`
  - `feature_importance.png`

## Conclusion
The project successfully demonstrates an advanced approach to fraud detection using ensemble learning and meta learning. The models achieved high accuracy, particularly with the meta learner approach, providing a robust solution for fraud detection.

---

This report presents a comprehensive view of the project results, methodologies, and insights gained during the analysis.
