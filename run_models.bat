@echo off
echo ========================================
echo Personality Prediction Model Runner
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python from https://python.org
    echo.
    echo After installing Python, run this script again.
    pause
    exit /b 1
)

echo Python found! Installing requirements...
pip install pandas numpy scikit-learn xgboost lightgbm

echo.
echo Running optimized model...
python optimized_model.py

echo.
echo Model execution completed!
echo Check for optimized_submission.csv in the current directory.
pause 