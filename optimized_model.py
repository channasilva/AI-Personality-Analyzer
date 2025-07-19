import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Separate features and target
X_train = train_data.drop(['id', 'Personality'], axis=1)
y_train = train_data['Personality']
X_test = test_data.drop(['id'], axis=1)

print("\nFeature columns:", X_train.columns.tolist())
print("Target distribution:")
print(y_train.value_counts())

# Advanced data preprocessing function
def advanced_preprocess_data(X_train, X_test):
    """Advanced preprocessing with multiple imputation strategies and feature engineering"""
    
    # Create copies
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Convert categorical variables
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    
    for col in categorical_cols:
        # Fill missing values with mode
        mode_val = X_train_processed[col].mode()[0] if not X_train_processed[col].mode().empty else 'No'
        X_train_processed[col] = X_train_processed[col].fillna(mode_val)
        X_test_processed[col] = X_test_processed[col].fillna(mode_val)
        
        # Encode categorical variables
        le = LabelEncoder()
        X_train_processed[col] = le.fit_transform(X_train_processed[col])
        X_test_processed[col] = le.transform(X_test_processed[col])
    
    # Handle numeric columns with KNN imputation for better accuracy
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency']
    
    # Use KNN imputation for numeric columns
    knn_imputer = KNNImputer(n_neighbors=5)
    X_train_processed[numeric_cols] = knn_imputer.fit_transform(X_train_processed[numeric_cols])
    X_test_processed[numeric_cols] = knn_imputer.transform(X_test_processed[numeric_cols])
    
    # Advanced feature engineering
    # 1. Social activity score (higher = more extroverted)
    X_train_processed['social_activity_score'] = (
        X_train_processed['Social_event_attendance'] + 
        X_train_processed['Going_outside'] + 
        X_train_processed['Post_frequency']
    )
    X_test_processed['social_activity_score'] = (
        X_test_processed['Social_event_attendance'] + 
        X_test_processed['Going_outside'] + 
        X_test_processed['Post_frequency']
    )
    
    # 2. Social energy indicator (lower = more introverted)
    X_train_processed['social_energy'] = (
        X_train_processed['Time_spent_Alone'] * 0.5 + 
        X_train_processed['Stage_fear'] * 2 + 
        X_train_processed['Drained_after_socializing'] * 2
    )
    X_test_processed['social_energy'] = (
        X_test_processed['Time_spent_Alone'] * 0.5 + 
        X_test_processed['Stage_fear'] * 2 + 
        X_test_processed['Drained_after_socializing'] * 2
    )
    
    # 3. Network size vs activity ratio
    X_train_processed['network_activity_ratio'] = (
        X_train_processed['Friends_circle_size'] / 
        (X_train_processed['Social_event_attendance'] + 1)
    )
    X_test_processed['network_activity_ratio'] = (
        X_test_processed['Friends_circle_size'] / 
        (X_test_processed['Social_event_attendance'] + 1)
    )
    
    # 4. Introversion indicators
    X_train_processed['introversion_score'] = (
        X_train_processed['Time_spent_Alone'] * 0.3 + 
        X_train_processed['Stage_fear'] * 1.5 + 
        X_train_processed['Drained_after_socializing'] * 1.5 - 
        X_train_processed['Social_event_attendance'] * 0.2
    )
    X_test_processed['introversion_score'] = (
        X_test_processed['Time_spent_Alone'] * 0.3 + 
        X_test_processed['Stage_fear'] * 1.5 + 
        X_test_processed['Drained_after_socializing'] * 1.5 - 
        X_test_processed['Social_event_attendance'] * 0.2
    )
    
    # 5. Extroversion indicators
    X_train_processed['extroversion_score'] = (
        X_train_processed['Social_event_attendance'] * 0.3 + 
        X_train_processed['Going_outside'] * 0.3 + 
        X_train_processed['Post_frequency'] * 0.2 + 
        X_train_processed['Friends_circle_size'] * 0.2 - 
        X_train_processed['Time_spent_Alone'] * 0.1
    )
    X_test_processed['extroversion_score'] = (
        X_test_processed['Social_event_attendance'] * 0.3 + 
        X_test_processed['Going_outside'] * 0.3 + 
        X_test_processed['Post_frequency'] * 0.2 + 
        X_test_processed['Friends_circle_size'] * 0.2 - 
        X_test_processed['Time_spent_Alone'] * 0.1
    )
    
    # 6. Social comfort zone
    X_train_processed['social_comfort'] = (
        X_train_processed['Friends_circle_size'] * 0.4 + 
        X_train_processed['Social_event_attendance'] * 0.3 + 
        X_train_processed['Going_outside'] * 0.3
    ) / (X_train_processed['Time_spent_Alone'] + 1)
    X_test_processed['social_comfort'] = (
        X_test_processed['Friends_circle_size'] * 0.4 + 
        X_test_processed['Social_event_attendance'] * 0.3 + 
        X_test_processed['Going_outside'] * 0.3
    ) / (X_test_processed['Time_spent_Alone'] + 1)
    
    # 7. Social engagement efficiency
    X_train_processed['social_efficiency'] = (
        X_train_processed['Friends_circle_size'] * X_train_processed['Social_event_attendance']
    ) / (X_train_processed['Time_spent_Alone'] + 1)
    X_test_processed['social_efficiency'] = (
        X_test_processed['Friends_circle_size'] * X_test_processed['Social_event_attendance']
    ) / (X_test_processed['Time_spent_Alone'] + 1)
    
    # 8. Digital vs physical social ratio
    X_train_processed['digital_physical_ratio'] = (
        X_train_processed['Post_frequency'] / 
        (X_train_processed['Social_event_attendance'] + X_train_processed['Going_outside'] + 1)
    )
    X_test_processed['digital_physical_ratio'] = (
        X_test_processed['Post_frequency'] / 
        (X_test_processed['Social_event_attendance'] + X_test_processed['Going_outside'] + 1)
    )
    
    # 9. Social anxiety composite
    X_train_processed['social_anxiety'] = (
        X_train_processed['Stage_fear'] * 2 + 
        X_train_processed['Drained_after_socializing'] * 2 + 
        X_train_processed['Time_spent_Alone'] * 0.3
    )
    X_test_processed['social_anxiety'] = (
        X_test_processed['Stage_fear'] * 2 + 
        X_test_processed['Drained_after_socializing'] * 2 + 
        X_test_processed['Time_spent_Alone'] * 0.3
    )
    
    # 10. Social confidence score
    X_train_processed['social_confidence'] = (
        X_train_processed['Social_event_attendance'] + 
        X_train_processed['Going_outside'] + 
        X_train_processed['Friends_circle_size'] * 0.5
    ) - (X_train_processed['Stage_fear'] + X_train_processed['Drained_after_socializing'])
    X_test_processed['social_confidence'] = (
        X_test_processed['Social_event_attendance'] + 
        X_test_processed['Going_outside'] + 
        X_test_processed['Friends_circle_size'] * 0.5
    ) - (X_test_processed['Stage_fear'] + X_test_processed['Drained_after_socializing'])
    
    # Handle infinite values
    X_train_processed = X_train_processed.replace([np.inf, -np.inf], np.nan)
    X_test_processed = X_test_processed.replace([np.inf, -np.inf], np.nan)
    
    # Fill any remaining NaN values
    X_train_processed = X_train_processed.fillna(0)
    X_test_processed = X_test_processed.fillna(0)
    
    return X_train_processed, X_test_processed

# Preprocess the data
print("\nPreprocessing data with advanced techniques...")
X_train_processed, X_test_processed = advanced_preprocess_data(X_train, X_test)

print(f"Processed training data shape: {X_train_processed.shape}")
print(f"Processed test data shape: {X_test_processed.shape}")
print("New features created:", [col for col in X_train_processed.columns if col not in X_train.columns])

# Robust scaling for better handling of outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_processed.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_processed.columns)

# Feature selection with RFE
print("\nPerforming advanced feature selection...")
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=base_model, n_features_to_select=15)
X_train_selected = rfe.fit_transform(X_train_scaled, y_train)
X_test_selected = rfe.transform(X_test_scaled)

# Get selected features
selected_features = X_train_scaled.columns[rfe.support_].tolist()
print(f"Selected {len(selected_features)} features using RFE")

# Use selected features
X_train_final = X_train_scaled[selected_features]
X_test_final = X_test_scaled[selected_features]

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Hyperparameter optimization for best models
print("\nPerforming hyperparameter optimization...")

# XGBoost optimization
xgb_param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train_split, y_train_split)
print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
print(f"Best XGBoost CV score: {xgb_grid.best_score_:.4f}")

# LightGBM optimization
lgb_param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgb_grid = GridSearchCV(lgb_model, lgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
lgb_grid.fit(X_train_split, y_train_split)
print(f"Best LightGBM parameters: {lgb_grid.best_params_}")
print(f"Best LightGBM CV score: {lgb_grid.best_score_:.4f}")

# Random Forest optimization
rf_param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [8, 10, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train_split, y_train_split)
print(f"Best Random Forest parameters: {rf_grid.best_params_}")
print(f"Best Random Forest CV score: {rf_grid.best_score_:.4f}")

# Create optimized ensemble
print("\nCreating optimized ensemble...")
optimized_models = [
    ('xgb_opt', xgb_grid.best_estimator_),
    ('lgb_opt', lgb_grid.best_estimator_),
    ('rf_opt', rf_grid.best_estimator_),
    ('et', ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42))
]

ensemble = VotingClassifier(
    estimators=optimized_models,
    voting='soft'
)

# Train ensemble
ensemble.fit(X_train_split, y_train_split)
ensemble_pred = ensemble.predict(X_val_split)
ensemble_accuracy = accuracy_score(y_val_split, ensemble_pred)
print(f"Optimized Ensemble Validation Accuracy: {ensemble_accuracy:.4f}")

# Cross-validation for final evaluation
print("\nPerforming cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ensemble_cv_scores = cross_val_score(ensemble, X_train_final, y_train, cv=cv, scoring='accuracy')
print(f"Optimized Ensemble CV Accuracy: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")

# Train final model on full training data
print("\nTraining final optimized model on full dataset...")
final_model = ensemble.fit(X_train_final, y_train)

# Make predictions on test set
print("Making predictions on test set...")
test_predictions = final_model.predict(X_test_final)

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'Personality': test_predictions
})

# Save submission
submission.to_csv('optimized_submission.csv', index=False)
print(f"\nOptimized submission saved with {len(submission)} predictions")

# Print prediction distribution
print("\nPrediction distribution:")
print(submission['Personality'].value_counts())

# Model comparison
print("\nModel Performance Summary:")
print("=" * 60)
print(f"XGBoost (Optimized): {xgb_grid.best_score_:.4f}")
print(f"LightGBM (Optimized): {lgb_grid.best_score_:.4f}")
print(f"Random Forest (Optimized): {rf_grid.best_score_:.4f}")
print(f"Ensemble (Optimized): {ensemble_accuracy:.4f}")
print(f"Cross-Validation Score: {ensemble_cv_scores.mean():.4f}")

print("\nFeature Importance (Selected Features):")
print("=" * 60)
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")

print(f"\nOptimized submission file 'optimized_submission.csv' created successfully!")
print(f"Expected accuracy: {ensemble_cv_scores.mean():.4f}")
print(f"Confidence interval: {ensemble_cv_scores.mean():.4f} ± {ensemble_cv_scores.std() * 2:.4f}") 