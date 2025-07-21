"""
Quick setup and run script for Advanced Fraud Detection
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def run_fraud_detection():
    """Run the fraud detection analysis"""
    try:
        print("\n🚀 Starting fraud detection analysis...")
        from fraud_detection_model import AdvancedFraudDetector
        
        # Initialize the fraud detector
        fraud_detector = AdvancedFraudDetector()
        
        # Run complete analysis with sample size for memory efficiency
        results = fraud_detector.run_complete_analysis(
            filepath='Fraud.csv',
            balance_method='smote',  # Options: 'smote', 'undersample', 'none'
            tune_hyperparameters=False  # Set to True for hyperparameter tuning (slower)
        )
        
        print("\n✅ Fraud detection analysis completed successfully!")
        print("📊 Check the generated plots: fraud_detection_evaluation.png and feature_importance.png")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running fraud detection: {e}")
        return None

if __name__ == "__main__":
    print("="*60)
    print("ADVANCED FRAUD DETECTION SYSTEM SETUP")
    print("="*60)
    
    # Check if Fraud.csv exists
    if not os.path.exists('Fraud.csv'):
        print("❌ Error: Fraud.csv file not found in current directory!")
        print("Please ensure the CSV file is in the same directory as this script.")
        sys.exit(1)
    
    # Install requirements
    if install_requirements():
        # Run fraud detection
        results = run_fraud_detection()
        
        if results:
            print("\n🎉 All done! Your fraud detection models are ready to use!")
        else:
            print("\n❌ Analysis failed. Please check the error messages above.")
    else:
        print("\n❌ Setup failed. Please install the required packages manually.")
