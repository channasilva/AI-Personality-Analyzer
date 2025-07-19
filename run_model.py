import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Packages installed successfully!")
    except subprocess.CalledProcessError:
        print("Error installing packages. Please install manually:")
        print("pip install pandas numpy scikit-learn xgboost lightgbm")
        return False
    return True

def run_model():
    """Run the optimized model"""
    print("\nRunning optimized personality prediction model...")
    try:
        # Import and run the optimized model
        from optimized_model import *
        print("\nModel execution completed successfully!")
        print("Check 'optimized_submission.csv' for predictions.")
    except Exception as e:
        print(f"Error running model: {e}")
        print("Trying basic model...")
        try:
            from personality_prediction_model import *
            print("\nBasic model execution completed successfully!")
            print("Check 'submission.csv' for predictions.")
        except Exception as e2:
            print(f"Error running basic model: {e2}")

if __name__ == "__main__":
    print("Personality Prediction Model")
    print("=" * 40)
    
    # Check if data files exist
    required_files = ['train.csv', 'test.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        print("Please ensure train.csv and test.csv are in the current directory.")
        sys.exit(1)
    
    # Install requirements
    if install_requirements():
        # Run the model
        run_model()
    else:
        print("Please install requirements manually and try again.") 