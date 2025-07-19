import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
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

# Encode target variable for XGBoost compatibility
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

print("\nFeature columns:", X_train.columns.tolist())
print("Target distribution:")
print(y_train.value_counts())
print("Encoded target classes:", label_encoder.classes_)

# Data preprocessing function
def preprocess_data(X_train, X_test):
    """Preprocess the data with advanced feature engineering"""
    
    # Create copies to avoid modifying original data
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
    
    # Handle numeric columns with missing values
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency']
    
    for col in numeric_cols:
        # Fill missing values with median
        median_val = X_train_processed[col].median()
        X_train_processed[col] = X_train_processed[col].fillna(median_val)
        X_test_processed[col] = X_test_processed[col].fillna(median_val)
    
    # Feature engineering
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
    
    # Handle infinite values
    X_train_processed = X_train_processed.replace([np.inf, -np.inf], np.nan)
    X_test_processed = X_test_processed.replace([np.inf, -np.inf], np.nan)
    
    # Fill any remaining NaN values
    X_train_processed = X_train_processed.fillna(0)
    X_test_processed = X_test_processed.fillna(0)
    
    return X_train_processed, X_test_processed

# Preprocess the data
print("\nPreprocessing data...")
X_train_processed, X_test_processed = preprocess_data(X_train, X_test)

print(f"Processed training data shape: {X_train_processed.shape}")
print(f"Processed test data shape: {X_test_processed.shape}")
print("New features created:", [col for col in X_train_processed.columns if col not in X_train.columns])

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# Convert back to DataFrame for better handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_processed.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_processed.columns)

# Feature selection
print("\nPerforming feature selection...")
selector = SelectKBest(score_func=f_classif, k='all')
X_train_selected = selector.fit_transform(X_train_scaled, y_train_encoded)
X_test_selected = selector.transform(X_test_scaled)

# Get feature scores
feature_scores = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)

print("Top 10 most important features:")
print(feature_scores.head(10))

# Use top features
top_features = feature_scores.head(10)['Feature'].tolist()
X_train_final = X_train_scaled[top_features]
X_test_final = X_test_scaled[top_features]

print(f"\nFinal feature set: {len(top_features)} features")

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_final, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

# Define models
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=5, 
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True
    ),
    'Logistic Regression': LogisticRegression(
        C=1.0,
        random_state=42,
        max_iter=1000
    )
}

# Train and evaluate individual models
print("\nTraining individual models...")
model_scores = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_split, y_train_split)
    val_pred = model.predict(X_val_split)
    accuracy = accuracy_score(y_val_split, val_pred)
    model_scores[name] = accuracy
    print(f"{name} Validation Accuracy: {accuracy:.4f}")

# Find best individual model
best_model_name = max(model_scores, key=model_scores.get)
print(f"\nBest individual model: {best_model_name} with accuracy: {model_scores[best_model_name]:.4f}")

# Create ensemble model
print("\nCreating ensemble model...")
ensemble_models = [
    ('rf', models['Random Forest']),
    ('gb', models['Gradient Boosting']),
    ('xgb', models['XGBoost']),
    ('lgb', models['LightGBM'])
]

ensemble = VotingClassifier(
    estimators=ensemble_models,
    voting='soft'
)

# Train ensemble
ensemble.fit(X_train_split, y_train_split)
ensemble_pred = ensemble.predict(X_val_split)
ensemble_accuracy = accuracy_score(y_val_split, ensemble_pred)
print(f"Ensemble Validation Accuracy: {ensemble_accuracy:.4f}")

# Cross-validation for final evaluation
print("\nPerforming cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validate ensemble
ensemble_cv_scores = cross_val_score(ensemble, X_train_final, y_train_encoded, cv=cv, scoring='accuracy')
print(f"Ensemble CV Accuracy: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")

# Train final model on full training data
print("\nTraining final model on full dataset...")
final_model = ensemble.fit(X_train_final, y_train_encoded)

# Make predictions on test set
print("Making predictions on test set...")
test_predictions_encoded = final_model.predict(X_test_final)

# Decode predictions back to original labels
test_predictions = label_encoder.inverse_transform(test_predictions_encoded)

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'Personality': test_predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved with {len(submission)} predictions")

# Print prediction distribution
print("\nPrediction distribution:")
print(submission['Personality'].value_counts())

# Additional analysis
print("\nModel Performance Summary:")
print("=" * 50)
for name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.4f}")
print(f"Ensemble: {ensemble_accuracy:.4f}")

print("\nFeature Importance (Top 10):")
print("=" * 50)
for i, (feature, score) in enumerate(zip(feature_scores['Feature'], feature_scores['Score'])):
    if i < 10:
        print(f"{i+1}. {feature}: {score:.2f}")

print(f"\nSubmission file 'submission.csv' created successfully!")
print(f"Expected accuracy: {ensemble_cv_scores.mean():.4f}") 