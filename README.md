# Advanced Fraud Detection System

This project implements a comprehensive fraud detection system using multiple machine learning algorithms and advanced feature engineering techniques.

## Features
- Data preprocessing and cleaning
- Advanced feature engineering
- Multiple ML models (Random Forest, XGBoost, Logistic Regression, Isolation Forest)
- Ensemble model with meta learner
- Comprehensive evaluation metrics
- Class imbalance handling

## Setup Instructions
1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Fraud Detection Analysis**
   Use the run script:
   ```
   python run_fraud_detection.py
   ```
   Or, run with a custom sample size:
   ```
   python run_with_sample.py <sample_size>
   ```

## Results (1,000,000 Samples)
- **Random Forest**: Accuracy 100.00%, F1-Score 99.91%, ROC-AUC 99.95%
- **XGBoost**: Accuracy 100.00%, F1-Score 99.85%, ROC-AUC 99.98%
- **Logistic Regression**: Accuracy 99.99%, F1-Score 99.60%, ROC-AUC 99.98% (**Best Performer**)
- **Meta Ensemble**: Accuracy 100.00%, F1-Score 99.85%, ROC-AUC 99.91%

## Top Important Features
1. `amount_equals_old_balance` (28.54%)
2. `step` (25.51%)
3. `balance_diff_orig` (11.74%)

## Visualization
- Evaluation plots and feature importance plots have been saved:
  - `fraud_detection_evaluation.png`
  - `feature_importance.png`

## Meta Learner Approach
Utilizes a Gradient Boosting Classifier to refine ensemble predictions for greater accuracy. The meta learner learns from ensemble predictions to provide a final probability score.

