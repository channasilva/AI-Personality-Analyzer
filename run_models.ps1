# Personality Prediction Model Runner
Write-Host "========================================" -ForegroundColor Green
Write-Host "Personality Prediction Model Runner" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if Python is available
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python from https://python.org" -ForegroundColor Red
    Write-Host "After installing Python, run this script again." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
try {
    pip install pandas numpy scikit-learn xgboost lightgbm
    Write-Host "Requirements installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "Error installing requirements. Please install manually:" -ForegroundColor Red
    Write-Host "pip install pandas numpy scikit-learn xgboost lightgbm" -ForegroundColor Red
}

Write-Host ""
Write-Host "Running optimized model..." -ForegroundColor Yellow
try {
    python optimized_model.py
    Write-Host ""
    Write-Host "Model execution completed!" -ForegroundColor Green
    Write-Host "Check for optimized_submission.csv in the current directory." -ForegroundColor Green
} catch {
    Write-Host "Error running model. Please check the error messages above." -ForegroundColor Red
}

Read-Host "Press Enter to exit" 