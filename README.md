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

## Results
- **Random Forest**: Accuracy 99.97%, F1-Score 99.94%, ROC-AUC 99.97%
- **XGBoost**: Accuracy 99.93%, F1-Score 99.88%, ROC-AUC 99.99%
- **Logistic Regression**: Accuracy 99.85%, F1-Score 99.73%, ROC-AUC 99.92%
- **Meta Ensemble**: Accuracy 99.93%, F1-Score 99.88%, ROC-AUC 99.94%

## Important Features
- Top feature: `step`

## Visualization
- Evaluation plots and feature importance plots have been saved:
  - `fraud_detection_evaluation.png`
  - `feature_importance.png`

## Meta Learner Approach
Utilizes a Gradient Boosting Classifier to refine ensemble predictions for greater accuracy. The meta learner learns from ensemble predictions to provide a final probability score.

