"""
Advanced Fraud Detection System
==============================

This script implements a comprehensive fraud detection system using multiple
machine learning algorithms and advanced feature engineering techniques.

Key Features:
- Data preprocessing and cleaning
- Advanced feature engineering
- Multiple ML models (Random Forest, XGBoost, Neural Network)
- Model ensemble techniques
- Comprehensive evaluation metrics
- Class imbalance handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class AdvancedFraudDetector:
    """
    Advanced Fraud Detection System with multiple ML algorithms and feature engineering
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.evaluation_results = {}
        
    def load_and_explore_data(self, filepath, sample_size=100000):
        """Load and perform initial exploration of the dataset"""
        print("Loading and exploring data...")
        
        # Load data with sampling for memory efficiency
        print(f"Loading a sample of {sample_size:,} rows for analysis...")
        
        # First, get the total number of rows
        total_rows = sum(1 for line in open(filepath)) - 1  # Subtract 1 for header
        print(f"Total rows in dataset: {total_rows:,}")
        
        if total_rows <= sample_size:
            # If dataset is small, load all data
            self.df = pd.read_csv(filepath)
            print("Loading entire dataset as it's within sample size.")
        else:
            # Sample data strategically to ensure we get fraud cases
            print("Sampling data to include fraud cases...")
            
            # Load data in chunks to find fraud cases
            fraud_samples = []
            normal_samples = []
            chunk_size = 10000
            fraud_target = min(sample_size // 10, 5000)  # At least 10% fraud cases
            normal_target = sample_size - fraud_target
            
            for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                # Separate fraud and normal transactions
                fraud_chunk = chunk[chunk['isFraud'] == 1]
                normal_chunk = chunk[chunk['isFraud'] == 0]
                
                # Add fraud samples if we need more
                if len(fraud_samples) < fraud_target and len(fraud_chunk) > 0:
                    needed_fraud = fraud_target - len(fraud_samples)
                    fraud_samples.append(fraud_chunk.head(needed_fraud))
                
                # Add normal samples if we need more
                if len(normal_samples) * chunk_size < normal_target and len(normal_chunk) > 0:
                    # Sample randomly from normal transactions
                    sample_size_chunk = min(len(normal_chunk), normal_target // 10)
                    normal_samples.append(normal_chunk.sample(n=sample_size_chunk, random_state=42))
                
                # Break if we have enough samples
                if (len(fraud_samples) >= fraud_target and 
                    sum(len(df) for df in normal_samples) >= normal_target):
                    break
            
            # Combine all samples
            all_samples = []
            if fraud_samples:
                all_samples.extend(fraud_samples)
            if normal_samples:
                all_samples.extend(normal_samples)
            
            if all_samples:
                self.df = pd.concat(all_samples, ignore_index=True)
                # Shuffle the combined dataset
                self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                # Fallback: just sample randomly
                print("Fallback: Random sampling...")
                skip_rows = sorted(np.random.choice(total_rows, total_rows - sample_size, replace=False))
                self.df = pd.read_csv(filepath, skiprows=skip_rows)
        
        # Basic info
        print(f"\nSampled dataset shape: {self.df.shape}")
        print(f"\nColumn types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
        # Fraud distribution
        fraud_distribution = self.df['isFraud'].value_counts()
        print(f"\nFraud distribution in sample:\n{fraud_distribution}")
        if len(fraud_distribution) > 1:
            print(f"Fraud percentage: {fraud_distribution[1] / len(self.df) * 100:.2f}%")
        else:
            print("No fraud cases found in sample")
        
        return self.df
    
    def advanced_feature_engineering(self):
        """Create advanced features for fraud detection"""
        print("Performing advanced feature engineering...")
        
        # Create a copy for feature engineering
        df_features = self.df.copy()
        
        # 1. Transaction amount features
        df_features['amount_log'] = np.log1p(df_features['amount'])
        df_features['amount_sqrt'] = np.sqrt(df_features['amount'])
        
        # 2. Balance features
        df_features['balance_diff_orig'] = df_features['oldbalanceOrg'] - df_features['newbalanceOrig']
        df_features['balance_diff_dest'] = df_features['newbalanceDest'] - df_features['oldbalanceDest']
        df_features['balance_ratio_orig'] = df_features['newbalanceOrig'] / (df_features['oldbalanceOrg'] + 1)
        df_features['balance_ratio_dest'] = df_features['newbalanceDest'] / (df_features['oldbalanceDest'] + 1)
        
        # 3. Transaction type encoding
        type_encoder = LabelEncoder()
        df_features['type_encoded'] = type_encoder.fit_transform(df_features['type'])
        self.label_encoders['type'] = type_encoder
        
        # 4. Anomaly indicators
        df_features['zero_balance_orig'] = (df_features['newbalanceOrig'] == 0).astype(int)
        df_features['zero_balance_dest'] = (df_features['newbalanceDest'] == 0).astype(int)
        df_features['amount_equals_old_balance'] = (df_features['amount'] == df_features['oldbalanceOrg']).astype(int)
        
        # 5. Statistical features by transaction type
        amount_stats = df_features.groupby('type')['amount'].agg(['mean', 'std']).reset_index()
        amount_stats.columns = ['type', 'type_amount_mean', 'type_amount_std']
        df_features = df_features.merge(amount_stats, on='type', how='left')
        
        df_features['amount_zscore'] = (df_features['amount'] - df_features['type_amount_mean']) / (df_features['type_amount_std'] + 1e-6)
        
        # 6. Time-based features (if step represents time)
        df_features['step_mod_24'] = df_features['step'] % 24  # Hour of day equivalent
        df_features['step_mod_7'] = df_features['step'] % 7    # Day of week equivalent
        
        # 7. Account activity features
        orig_activity = df_features.groupby('nameOrig').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'step': 'nunique'
        }).fillna(0)
        orig_activity.columns = ['orig_trans_count', 'orig_total_amount', 'orig_avg_amount', 'orig_amount_std', 'orig_active_steps']
        
        dest_activity = df_features.groupby('nameDest').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'step': 'nunique'
        }).fillna(0)
        dest_activity.columns = ['dest_trans_count', 'dest_total_amount', 'dest_avg_amount', 'dest_amount_std', 'dest_active_steps']
        
        df_features = df_features.merge(orig_activity, left_on='nameOrig', right_index=True, how='left')
        df_features = df_features.merge(dest_activity, left_on='nameDest', right_index=True, how='left')
        
        # Fill NaN values
        df_features = df_features.fillna(0)
        
        # Select numerical features for modeling
        feature_columns = [
            'step', 'type_encoded', 'amount', 'amount_log', 'amount_sqrt',
            'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'balance_diff_orig', 'balance_diff_dest', 'balance_ratio_orig', 'balance_ratio_dest',
            'zero_balance_orig', 'zero_balance_dest', 'amount_equals_old_balance',
            'type_amount_mean', 'type_amount_std', 'amount_zscore',
            'step_mod_24', 'step_mod_7',
            'orig_trans_count', 'orig_total_amount', 'orig_avg_amount', 'orig_amount_std', 'orig_active_steps',
            'dest_trans_count', 'dest_total_amount', 'dest_avg_amount', 'dest_amount_std', 'dest_active_steps'
        ]
        
        self.X = df_features[feature_columns]
        self.y = df_features['isFraud']
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Features created: {len(feature_columns)}")
        
        return self.X, self.y
    
    def handle_class_imbalance(self, method='smote'):
        """Handle class imbalance using various techniques"""
        print(f"Handling class imbalance using {method}...")
        
        if method == 'smote':
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(self.X_train, self.y_train)
        else:  # no resampling
            X_resampled, y_resampled = self.X_train, self.y_train
        
        print(f"Original training set: {len(self.y_train)} samples")
        print(f"Resampled training set: {len(y_resampled)} samples")
        print(f"New fraud ratio: {y_resampled.sum() / len(y_resampled):.3f}")
        
        return X_resampled, y_resampled
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data and prepare for training"""
        print("Preparing data for training...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self, X_train, y_train, tune_params=True):
        """Train Random Forest model with optional hyperparameter tuning"""
        print("Training Random Forest model...")
        
        if tune_params:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            self.models['random_forest'] = grid_search.best_estimator_
            print(f"Best RF parameters: {grid_search.best_params_}")
        else:
            rf = RandomForestClassifier(
                n_estimators=200, 
                max_depth=20, 
                random_state=42, 
                class_weight='balanced',
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            self.models['random_forest'] = rf
        
        # Feature importance
        self.feature_importance['random_forest'] = self.models['random_forest'].feature_importances_
    
    def train_xgboost(self, X_train, y_train, tune_params=True):
        """Train XGBoost model with optional hyperparameter tuning"""
        print("Training XGBoost model...")
        
        if tune_params:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            xgb_model = xgb.XGBClassifier(
                random_state=42, 
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )
            
            grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            self.models['xgboost'] = grid_search.best_estimator_
            print(f"Best XGB parameters: {grid_search.best_params_}")
        else:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model
        
        # Feature importance
        self.feature_importance['xgboost'] = self.models['xgboost'].feature_importances_
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("Training Logistic Regression model...")
        
        lr = LogisticRegression(
            random_state=42, 
            class_weight='balanced', 
            max_iter=1000,
            solver='liblinear'
        )
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
    
    def train_isolation_forest(self, X_train):
        """Train Isolation Forest for anomaly detection"""
        print("Training Isolation Forest model...")
        
        # Isolation Forest is an unsupervised method
        iso_forest = IsolationForest(
            contamination=0.1,  # Expected proportion of outliers
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_train)
        self.models['isolation_forest'] = iso_forest
    
    def create_ensemble_model(self, X_train, y_train):
        """Create ensemble model combining predictions"""
        print("Creating ensemble model...")
        
        # Get predictions from individual models
        rf_pred = self.models['random_forest'].predict_proba(X_train)[:, 1]
        xgb_pred = self.models['xgboost'].predict_proba(X_train)[:, 1]
        lr_pred = self.models['logistic_regression'].predict_proba(X_train)[:, 1]
        
        # Weighted ensemble (you can adjust weights based on individual model performance)
        ensemble_pred = 0.4 * rf_pred + 0.4 * xgb_pred + 0.2 * lr_pred
        
        self.ensemble_weights = {'rf': 0.4, 'xgb': 0.4, 'lr': 0.2}
        
        return ensemble_pred
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a specific model"""
        print(f"Evaluating {model_name} model...")
        
        model = self.models[model_name]
        
        if model_name == 'isolation_forest':
            # Isolation Forest returns -1 for anomalies, 1 for normal
            predictions = model.predict(X_test)
            y_pred = (predictions == -1).astype(int)
            y_pred_proba = model.score_samples(X_test)
            # Convert scores to probabilities (higher anomaly score = higher fraud probability)
            y_pred_proba = 1 / (1 + np.exp(y_pred_proba))
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        self.evaluation_results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{model_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.evaluation_results[model_name]
    
    def train_meta_learner(self, X_meta, y_meta):
        """Train a meta learner to produce final probability score"""
        print("Training Meta Learner...")

        # Using a Gradient Boosting Classifier as a meta learner
        meta_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        meta_model.fit(X_meta, y_meta)
        self.models['meta_learner'] = meta_model

    def create_ensemble_model_with_meta(self, X_train, y_train):
        """Create ensemble model with a meta learner combining predictions"""
        print("Creating ensemble model with meta learner...")

        # Prepare meta-features
        rf_pred = self.models['random_forest'].predict_proba(X_train)[:, 1]
        xgb_pred = self.models['xgboost'].predict_proba(X_train)[:, 1]
        lr_pred = self.models['logistic_regression'].predict_proba(X_train)[:, 1]

        # Create meta feature set
        X_meta = np.column_stack((rf_pred, xgb_pred, lr_pred))

        # Train meta learner
        self.train_meta_learner(X_meta, y_train)

    def evaluate_ensemble_with_meta(self, X_test, y_test):
        """Evaluate ensemble model with meta learner"""
        print("Evaluating ensemble model with meta learner...")

        # Prepare meta-features
        rf_pred = self.models['random_forest'].predict_proba(X_test)[:, 1]
        xgb_pred = self.models['xgboost'].predict_proba(X_test)[:, 1]
        lr_pred = self.models['logistic_regression'].predict_proba(X_test)[:, 1]

        # Create meta feature set
        X_meta_test = np.column_stack((rf_pred, xgb_pred, lr_pred))

        # Predict with meta learner
        meta_pred_proba = self.models['meta_learner'].predict_proba(X_meta_test)[:, 1]
        meta_pred = (meta_pred_proba > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, meta_pred)
        f1 = f1_score(y_test, meta_pred)
        roc_auc = roc_auc_score(y_test, meta_pred_proba)

        self.evaluation_results['meta_ensemble'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': meta_pred,
            'probabilities': meta_pred_proba
        }

        print(f"Meta Learner Ensemble Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")

        return self.evaluation_results['meta_ensemble']
    
    def plot_evaluation_results(self):
        """Plot comprehensive evaluation results"""
        print("Generating evaluation plots...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Model comparison
        models = ['random_forest', 'xgboost', 'logistic_regression', 'isolation_forest', 'ensemble']
        metrics = ['accuracy', 'f1_score', 'roc_auc']
        
        comparison_data = []
        for model in models:
            if model in self.evaluation_results:
                for metric in metrics:
                    comparison_data.append({
                        'Model': model,
                        'Metric': metric,
                        'Score': self.evaluation_results[model][metric]
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot model comparison
        for i, metric in enumerate(metrics):
            metric_data = comparison_df[comparison_df['Metric'] == metric]
            axes[0, i].bar(metric_data['Model'], metric_data['Score'])
            axes[0, i].set_title(f'{metric.upper()} Comparison')
            axes[0, i].tick_params(axis='x', rotation=45)
            axes[0, i].set_ylim([0, 1])
        
        # 2. ROC Curves
        for model_name in ['random_forest', 'xgboost', 'logistic_regression', 'ensemble']:
            if model_name in self.evaluation_results:
                fpr, tpr, _ = roc_curve(self.y_test, self.evaluation_results[model_name]['probabilities'])
                axes[1, 0].plot(fpr, tpr, label=f'{model_name} (AUC: {self.evaluation_results[model_name]["roc_auc"]:.3f})')
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curves')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        for model_name in ['random_forest', 'xgboost', 'logistic_regression', 'ensemble']:
            if model_name in self.evaluation_results:
                precision, recall, _ = precision_recall_curve(self.y_test, self.evaluation_results[model_name]['probabilities'])
                axes[1, 1].plot(recall, precision, label=f'{model_name}')
        
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix for best model (ensemble)
        if 'ensemble' in self.evaluation_results:
            cm = confusion_matrix(self.y_test, self.evaluation_results['ensemble']['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2])
            axes[1, 2].set_title('Confusion Matrix (Ensemble)')
            axes[1, 2].set_xlabel('Predicted')
            axes[1, 2].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('fraud_detection_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        print("Plotting feature importance...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        feature_names = self.X.columns
        
        # Random Forest feature importance
        if 'random_forest' in self.feature_importance:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance['random_forest']
            }).sort_values('importance', ascending=True).tail(15)
            
            axes[0].barh(importance_df['feature'], importance_df['importance'])
            axes[0].set_title('Random Forest Feature Importance')
            axes[0].set_xlabel('Importance')
        
        # XGBoost feature importance
        if 'xgboost' in self.feature_importance:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance['xgboost']
            }).sort_values('importance', ascending=True).tail(15)
            
            axes[1].barh(importance_df['feature'], importance_df['importance'])
            axes[1].set_title('XGBoost Feature Importance')
            axes[1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_fraud_report(self):
        """Generate comprehensive fraud detection report"""
        print("\n" + "="*60)
        print("ADVANCED FRAUD DETECTION SYSTEM - FINAL REPORT")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"- Total transactions: {len(self.df):,}")
        print(f"- Fraudulent transactions: {self.df['isFraud'].sum():,}")
        print(f"- Fraud rate: {self.df['isFraud'].mean()*100:.2f}%")
        print(f"- Features engineered: {self.X.shape[1]}")
        
        print(f"\nModel Performance Summary:")
        print("-" * 40)
        
        for model_name, results in self.evaluation_results.items():
            print(f"{model_name.upper():20} | "
                  f"Accuracy: {results['accuracy']:.4f} | "
                  f"F1: {results['f1_score']:.4f} | "
                  f"AUC: {results['roc_auc']:.4f}")
        
        # Find best model
        best_model = max(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x]['roc_auc'])
        
        print(f"\nBest Performing Model: {best_model.upper()}")
        print(f"- ROC-AUC Score: {self.evaluation_results[best_model]['roc_auc']:.4f}")
        print(f"- F1 Score: {self.evaluation_results[best_model]['f1_score']:.4f}")
        print(f"- Accuracy: {self.evaluation_results[best_model]['accuracy']:.4f}")
        
        print(f"\nTop 10 Most Important Features:")
        if 'random_forest' in self.feature_importance:
            feature_names = self.X.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance['random_forest']
            }).sort_values('importance', ascending=False).head(10)
            
            for i, (_, row) in enumerate(importance_df.iterrows(), 1):
                print(f"{i:2d}. {row['feature']:25} ({row['importance']:.4f})")
        
        print("\n" + "="*60)
    
    def run_complete_analysis(self, filepath, balance_method='smote', tune_hyperparameters=False):
        """Run the complete fraud detection analysis pipeline"""
        print("Starting Advanced Fraud Detection Analysis...")
        
        # 1. Load and explore data
        self.load_and_explore_data(filepath)
        
        # 2. Feature engineering
        self.advanced_feature_engineering()
        
        # 3. Prepare data
        self.prepare_data()
        
        # 4. Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(balance_method)
        
        # 5. Train models
        self.train_random_forest(X_train_balanced, y_train_balanced, tune_hyperparameters)
        self.train_xgboost(X_train_balanced, y_train_balanced, tune_hyperparameters)
        self.train_logistic_regression(self.X_train_scaled, self.y_train)
        self.train_isolation_forest(self.X_train_scaled)
        
        # 6. Create ensemble
        self.create_ensemble_model(self.X_train, self.y_train)
        
        # 7. Evaluate models
        self.evaluate_model('random_forest', self.X_test, self.y_test)
        self.evaluate_model('xgboost', self.X_test, self.y_test)
        self.evaluate_model('logistic_regression', self.X_test_scaled, self.y_test)
        self.evaluate_model('isolation_forest', self.X_test_scaled, self.y_test)
        self.evaluate_ensemble(self.X_test, self.y_test)
        
        # 8. Generate visualizations
        self.plot_evaluation_results()
        self.plot_feature_importance()
        
        # 9. Generate report
        self.generate_fraud_report()
        
        print("\nAnalysis completed successfully!")
        return self.evaluation_results

# Usage example
if __name__ == "__main__":
    # Initialize the fraud detector
    fraud_detector = AdvancedFraudDetector()
    
    # Run complete analysis
    results = fraud_detector.run_complete_analysis(
        filepath='Fraud.csv',
        balance_method='smote',  # Options: 'smote', 'undersample', 'none'
        tune_hyperparameters=False  # Set to True for hyperparameter tuning (slower)
    )
