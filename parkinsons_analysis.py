import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import joblib

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Parkinson's Disease Detection Using Voice Analysis")
print("=" * 60)
print("Student Project - ML Course")
print("Using UCI Parkinson's Dataset")
print()

# Load the dataset
df = pd.read_csv('parkinsons.data')
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")

# Quick data exploration
print(f"\nClass distribution:")
print(df['status'].value_counts())
print(f"Class balance: {df['status'].value_counts(normalize=True)}")

print("\nDataset info:")
print(df.info())

print("\nFirst few rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Visualize class distribution
plt.figure(figsize=(8, 6))
class_counts = df['status'].value_counts()
plt.pie(class_counts.values, labels=['Healthy', 'Parkinson\'s'], autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution in Dataset')
plt.axis('equal')
plt.show()

# Prepare features and target
X = df.drop(['name', 'status'], axis=1)  # Remove name column and target
y = df['status']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features for SVM and Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully!")

# Define models to test
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

# Store results
results = {}
trained_models = {}

print("\nTraining models...")
print("=" * 50)

# Train each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Random Forest doesn't need scaling
    if name == 'Random Forest':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # SVM and Logistic Regression need scaling
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    trained_models[name] = model
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

print("\nAll models trained successfully!")

# Compare results
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
    'Precision': [results[model]['precision'] for model in results.keys()],
    'Recall': [results[model]['recall'] for model in results.keys()],
    'F1-Score': [results[model]['f1'] for model in results.keys()],
    'ROC-AUC': [results[model]['roc_auc'] for model in results.keys()]
})

print("\nModel Performance Comparison:")
print("=" * 60)
print(results_df.round(4))

# Find best model
best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
print(f"\nBest performing model: {best_model_name}")
print(f"Best ROC-AUC: {results_df['ROC-AUC'].max():.4f}")

# Create performance visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy comparison
axes[0, 0].bar(results_df['Model'], results_df['Accuracy'], color='skyblue')
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].tick_params(axis='x', rotation=45)

# ROC-AUC comparison
axes[0, 1].bar(results_df['Model'], results_df['ROC-AUC'], color='lightcoral')
axes[0, 1].set_title('Model ROC-AUC Comparison')
axes[0, 1].set_ylabel('ROC-AUC')
axes[0, 1].tick_params(axis='x', rotation=45)

# F1-Score comparison
axes[1, 0].bar(results_df['Model'], results_df['F1-Score'], color='lightgreen')
axes[1, 0].set_title('Model F1-Score Comparison')
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].tick_params(axis='x', rotation=45)

# All metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(results_df))
width = 0.15

for i, metric in enumerate(metrics):
    axes[1, 1].bar(x + i*width, results_df[metric], width, label=metric)

axes[1, 1].set_title('All Metrics Comparison')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_xlabel('Models')
axes[1, 1].set_xticks(x + width * 2)
axes[1, 1].set_xticklabels(results_df['Model'], rotation=45)
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# ROC curves
plt.figure(figsize=(10, 8))

for name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Feature importance analysis
rf_model = trained_models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features (Random Forest):")
print("=" * 50)
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Cross-validation analysis
print("\nCross-Validation Results:")
print("=" * 50)

cv_scores = {}

for name, model in models.items():
    print(f"\n{name} Cross-Validation:")
    
    if name == 'Random Forest':
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    else:
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    cv_scores[name] = scores
    print(f"CV Scores: {scores}")
    print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Cross-validation visualization
plt.figure(figsize=(10, 6))
plt.boxplot([cv_scores[name] for name in cv_scores.keys()], labels=list(cv_scores.keys()))
plt.title('Cross-Validation ROC-AUC Scores')
plt.ylabel('ROC-AUC Score')
plt.grid(True, alpha=0.3)
plt.show()

# Final model selection
print("\nFinal Model Selection")
print("=" * 50)

best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_model = trained_models[best_model_name]
best_results = results[best_model_name]

print(f"Selected Model: {best_model_name}")
print(f"Final Performance:")
print(f"  Accuracy: {best_results['accuracy']:.4f}")
print(f"  Precision: {best_results['precision']:.4f}")
print(f"  Recall: {best_results['recall']:.4f}")
print(f"  F1-Score: {best_results['f1']:.4f}")
print(f"  ROC-AUC: {best_results['roc_auc']:.4f}")

# Save the best model for deployment
joblib.dump(best_model, 'best_parkinsons_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(X.columns, 'feature_names.pkl')

print("\nModel and preprocessing objects saved successfully!")
print("Files created:")
print("  - best_parkinsons_model.pkl")
print("  - feature_scaler.pkl")
print("  - feature_names.pkl")

# Summary of findings
print("\nKey Findings and Insights")
print("=" * 50)

print(f"\n1. Dataset Characteristics:")
print(f"   - Total samples: {len(df)}")
print(f"   - Features: {df.shape[1]-1}")
print(f"   - Class balance: {df['status'].value_counts(normalize=True)[0]:.1%} healthy, {df['status'].value_counts(normalize=True)[1]:.1%} Parkinson's")

print(f"\n2. Model Performance:")
print(f"   - Best model: {best_model_name}")
print(f"   - Best ROC-AUC: {best_results['roc_auc']:.4f}")
print(f"   - Best Accuracy: {best_results['accuracy']:.4f}")

print(f"\n3. Most Important Features:")
top_5_features = feature_importance.head(5)
for idx, row in top_5_features.iterrows():
    print(f"   - {row['feature']}: {row['importance']:.4f}")

print(f"\n4. Clinical Relevance:")
print("   - Voice-based biomarkers show promise for Parkinson's detection")
print("   - Jitter and shimmer measures are particularly important")
print("   - Non-linear features (RPDE, DFA) contribute significantly")

print(f"\n5. Model Limitations:")
print("   - Dataset size is relatively small")
print("   - Cross-validation shows some variance in performance")
print("   - Need more diverse population samples for generalization")

print(f"\n6. Future Work:")
print("   - Collect more diverse data")
print("   - Try deep learning approaches")
print("   - Feature engineering with domain knowledge")
print("   - Ensemble methods")
print("   - Real-time voice analysis implementation")

print("\n" + "="*60)
print("Project completed successfully!")
print("You can now run the Streamlit app with: streamlit run app.py")
print("="*60)
