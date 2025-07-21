"""
Run fraud detection with custom sample size
"""

import sys
from fraud_detection_model import AdvancedFraudDetector

def main():
    print("="*60)
    print("FRAUD DETECTION WITH CUSTOM SAMPLE SIZE")
    print("="*60)
    
    # Get sample size from command line or use default
    sample_size = 2000000  # Maximum sample size for GPU utilization
    
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
            print(f"Using custom sample size: {sample_size:,}")
        except ValueError:
            print("Invalid sample size provided. Using default: 50,000")
    else:
        print(f"Using default sample size: {sample_size:,}")
        print("Usage: python run_with_sample.py [sample_size]")
        print("Example: python run_with_sample.py 25000")
    
    print("\n🚀 Starting fraud detection analysis...")
    
    try:
        # Initialize the fraud detector
        fraud_detector = AdvancedFraudDetector()
        
        # Load data with custom sample size
        fraud_detector.load_and_explore_data('Fraud.csv', sample_size=sample_size)
        
        # Feature engineering
        fraud_detector.advanced_feature_engineering()
        
        # Prepare data
        fraud_detector.prepare_data()
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = fraud_detector.handle_class_imbalance('smote')
        
        # Train models (simplified for faster execution)
        print("\nTraining models...")
        fraud_detector.train_random_forest(X_train_balanced, y_train_balanced, tune_params=False)
        fraud_detector.train_xgboost(X_train_balanced, y_train_balanced, tune_params=False)
        fraud_detector.train_logistic_regression(fraud_detector.X_train_scaled, fraud_detector.y_train)
        
        # Create ensemble
        fraud_detector.create_ensemble_model_with_meta(fraud_detector.X_train, fraud_detector.y_train)
        
        # Evaluate models
        print("\nEvaluating models...")
        fraud_detector.evaluate_model('random_forest', fraud_detector.X_test, fraud_detector.y_test)
        fraud_detector.evaluate_model('xgboost', fraud_detector.X_test, fraud_detector.y_test)
        fraud_detector.evaluate_model('logistic_regression', fraud_detector.X_test_scaled, fraud_detector.y_test)
        fraud_detector.evaluate_ensemble_with_meta(fraud_detector.X_test, fraud_detector.y_test)
        
        # Generate report
        fraud_detector.generate_fraud_report()
        
        # Generate plots
        try:
            fraud_detector.plot_evaluation_results()
            fraud_detector.plot_feature_importance()
            print("\n📊 Plots saved: fraud_detection_evaluation.png and feature_importance.png")
        except Exception as e:
            print(f"⚠️  Could not generate plots: {e}")
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("\nTry reducing the sample size if you're running out of memory.")
        print("Example: python run_with_sample.py 25000")

if __name__ == "__main__":
    main()
