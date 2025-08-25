#!/usr/bin/env python3
"""
Cardiovascular Disease Prediction - ML Model Implementation
Implements SVM, KNN, Decision Tree, Logistic Regression, and Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load the processed cardiovascular dataset and prepare features"""
    print("Loading processed cardiovascular dataset...")
    
    # Try to load processed data, fallback to original if needed
    try:
        df = pd.read_csv('processed_cardio.csv')
        print(f"Loaded processed dataset with {len(df)} records")
    except FileNotFoundError:
        print("Processed file not found, loading original dataset...")
        df = pd.read_csv('src/data/cardio_train.csv', delimiter=';')
        print(f"Loaded original dataset with {len(df)} records")
        
        # Basic preprocessing if using original data
        df = preprocess_original_data(df)
    
    return df

def preprocess_original_data(df):
    """Preprocess original data if processed version not available"""
    print("Performing basic preprocessing on original data...")
    
    # Remove obvious outliers
    df = df[
        (df['age'] >= 30*365) & (df['age'] <= 80*365) &  # Age in days
        (df['height'] >= 140) & (df['height'] <= 210) &
        (df['weight'] >= 30) & (df['weight'] <= 200) &
        (df['ap_hi'] >= 70) & (df['ap_hi'] <= 250) &
        (df['ap_lo'] >= 40) & (df['ap_lo'] <= 150) &
        (df['ap_hi'] > df['ap_lo'])  # Systolic > Diastolic
    ]
    
    # Remove extreme BMI values
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 50)]
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    print(f"After preprocessing: {len(df)} records")
    return df

def prepare_features(df):
    """Prepare features for machine learning"""
    print("Preparing features for machine learning...")
    
    # Select features for modeling
    feature_columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                      'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    
    # Ensure all required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[feature_columns].copy()
    y = df['cardio'].copy()
    
    # Convert age from days to years for better interpretability
    if 'age' in X.columns:
        X['age'] = X['age'] / 365.25
    
    # Create additional features
    if all(col in X.columns for col in ['weight', 'height']):
        X['bmi'] = X['weight'] / ((X['height'] / 100) ** 2)
    
    if all(col in X.columns for col in ['ap_hi', 'ap_lo']):
        X['pulse_pressure'] = X['ap_hi'] - X['ap_lo']
        X['mean_arterial_pressure'] = X['ap_lo'] + X['pulse_pressure'] / 3
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def initialize_models():
    """Initialize all machine learning models"""
    models = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a single model and return metrics"""
    print(f"\nTraining {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # ROC-AUC (only if probabilities available)
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = roc_auc_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV_Mean': cv_mean,
        'CV_Std': cv_std,
        'Trained_Model': model
    }

def main():
    """Main execution function"""
    print("=" * 60)
    print("CARDIOVASCULAR DISEASE PREDICTION - ML MODEL EVALUATION")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    X, y = prepare_features(df)
    
    # Split the data
    print(f"\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features for SVM, KNN, and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = initialize_models()
    
    # Evaluate each model
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        # Use scaled data for models that benefit from it
        if model_name in ['SVM', 'KNN', 'Logistic Regression']:
            result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, model_name)
        else:
            result = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
        
        results.append(result)
        trained_models[model_name] = result['Trained_Model']
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.drop('Trained_Model', axis=1)
    
    # Sort by accuracy (descending)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(results_df.round(4).to_string(index=False))
    
    # Find best model
    best_model_name = results_df.iloc[0]['Model']
    best_accuracy = results_df.iloc[0]['Accuracy']
    best_model = trained_models[best_model_name]
    
    print(f"\n" + "=" * 60)
    print(f"BEST PERFORMING MODEL: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print("=" * 60)
    
    # Save the best model
    model_data = {
        'model': best_model,
        'scaler': scaler if best_model_name in ['SVM', 'KNN', 'Logistic Regression'] else None,
        'feature_names': list(X.columns),
        'model_name': best_model_name,
        'performance_metrics': results_df.iloc[0].to_dict()
    }
    
    with open('final_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nBest model saved as 'final_model.pkl'")
    
    # Save detailed results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("Detailed results saved as 'model_comparison_results.csv'")
    
    # Print detailed classification report for best model
    print(f"\n" + "=" * 60)
    print(f"DETAILED CLASSIFICATION REPORT - {best_model_name}")
    print("=" * 60)
    
    if best_model_name in ['SVM', 'KNN', 'Logistic Regression']:
        y_pred_best = best_model.predict(X_test_scaled)
    else:
        y_pred_best = best_model.predict(X_test)
    
    print(classification_report(y_test, y_pred_best, target_names=['No Disease', 'Has Disease']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                No    Yes")
    print(f"Actual No    {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"       Yes   {cm[1,0]:5d} {cm[1,1]:5d}")
    
    print(f"\nProcessing completed successfully!")
    print(f"Files generated:")
    print(f"- final_model.pkl (Best model: {best_model_name})")
    print(f"- model_comparison_results.csv (Detailed metrics)")

if __name__ == "__main__":
    main()