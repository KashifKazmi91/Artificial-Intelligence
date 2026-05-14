"""
Experiment 2: Logistic Regression

This script demonstrates logistic regression for binary classification.
We'll predict whether an email is spam or not.

Conceptual Flow:
1. Load and explore data (check class distribution)
2. Prepare features and target
3. Split into training and testing
4. Train logistic regression model
5. Make probability and class predictions
6. Evaluate with multiple metrics
7. Visualize confusion matrix and ROC curve
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve,
    classification_report, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EXPERIMENT 2: LOGISTIC REGRESSION")
print("Binary Classification: Email Spam Detection")
print("="*70)

# Step 1: Create Sample Dataset
print("\n[STEP 1] Creating Sample Dataset...")

np.random.seed(42)

# Features for each email
n_samples = 200

# Feature 1: Number of words in email
word_count = np.concatenate([
    np.random.normal(150, 50, n_samples//2),  # Not spam: 150 words avg
    np.random.normal(300, 80, n_samples//2)   # Spam: 300 words avg
])

# Feature 2: Number of links
link_count = np.concatenate([
    np.random.poisson(2, n_samples//2),       # Not spam: 2 links avg
    np.random.poisson(8, n_samples//2)        # Spam: 8 links avg
])

# Feature 3: Contains capital letters percentage
caps_ratio = np.concatenate([
    np.random.beta(5, 15, n_samples//2) * 100,  # Not spam: lower caps
    np.random.beta(15, 5, n_samples//2) * 100   # Spam: higher caps
])

# Target: is_spam (0 = Not spam, 1 = Spam)
is_spam = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)]).astype(int)

# Shuffle the data
indices = np.random.permutation(n_samples)
X_raw = np.column_stack([word_count, link_count, caps_ratio])[indices]
y = is_spam[indices]

# Create DataFrame
data = pd.DataFrame({
    'word_count': X_raw[:, 0],
    'link_count': X_raw[:, 1],
    'caps_ratio': X_raw[:, 2],
    'is_spam': y
})

print("\nDataset Preview:")
print(data.head(10))
print(f"\nDataset Shape: {data.shape}")
print(f"\nClass Distribution:")
print(data['is_spam'].value_counts())
print(f"\nClass Distribution (%)")
print(data['is_spam'].value_counts(normalize=True) * 100)

print(f"\nDataset Statistics:")
print(data.describe())

# Step 2: Prepare Data
print("\n" + "="*70)
print("[STEP 2] Preparing Data...")
print("="*70)

# Features (what we use to predict)
X = data[['word_count', 'link_count', 'caps_ratio']].values

# Target (what we want to predict)
y = data['is_spam'].values

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeature names: {['word_count', 'link_count', 'caps_ratio']}")

# Step 3: Scale Features (Important for Logistic Regression)
print("\n" + "="*70)
print("[STEP 3] Scaling Features...")
print("="*70)

print("\nBefore scaling - Feature ranges:")
print(f"  Word count: {X[:, 0].min():.1f} to {X[:, 0].max():.1f}")
print(f"  Link count: {X[:, 1].min():.1f} to {X[:, 1].max():.1f}")
print(f"  Caps ratio: {X[:, 2].min():.1f} to {X[:, 2].max():.1f}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nAfter scaling - Feature means and stds:")
print(f"  Word count: mean={X_scaled[:, 0].mean():.3f}, std={X_scaled[:, 0].std():.3f}")
print(f"  Link count: mean={X_scaled[:, 1].mean():.3f}, std={X_scaled[:, 1].std():.3f}")
print(f"  Caps ratio: mean={X_scaled[:, 2].mean():.3f}, std={X_scaled[:, 2].std():.3f}")

print("\n✓ Features scaled! All features now have mean≈0 and std≈1")

# Step 4: Train-Test Split
print("\n" + "="*70)
print("[STEP 4] Splitting Data...")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)} samples (80%)")
print(f"Testing set size: {len(X_test)} samples (20%)")

print(f"\nTraining set class distribution:")
print(f"  Not spam (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Spam (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

print(f"\nTesting set class distribution:")
print(f"  Not spam (0): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"  Spam (1): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

# Step 5: Create and Train Model
print("\n" + "="*70)
print("[STEP 5] Training Logistic Regression Model...")
print("="*70)

model = LogisticRegression(random_state=42, max_iter=1000)
print("\nTraining model...")
model.fit(X_train, y_train)
print("✓ Model trained successfully!")

print(f"\nModel Parameters:")
print(f"  Intercept: {model.intercept_[0]:.4f}")
print(f"  Coefficients: {model.coef_[0]}")
print(f"    - Word count coefficient: {model.coef_[0][0]:.4f}")
print(f"    - Link count coefficient: {model.coef_[0][1]:.4f}")
print(f"    - Caps ratio coefficient: {model.coef_[0][2]:.4f}")

print(f"\nCoefficient Interpretation:")
print(f"  + means increases spam probability")
print(f"  - means decreases spam probability")
for i, feature in enumerate(['word_count', 'link_count', 'caps_ratio']):
    coef = model.coef_[0][i]
    direction = "increases" if coef > 0 else "decreases"
    print(f"  • {feature}: {direction} spam probability (|coef|={abs(coef):.4f})")

# Step 6: Make Predictions
print("\n" + "="*70)
print("[STEP 6] Making Predictions...")
print("="*70)

# Get probability predictions
y_train_proba = model.predict_proba(X_train)
y_test_proba = model.predict_proba(X_test)

# Get class predictions (default threshold = 0.5)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"\nSample Predictions on Test Set:")
print("\nIndex | Actual | Prob(Not Spam) | Prob(Spam) | Predicted | Correct")
print("-" * 70)
for i in range(min(10, len(X_test))):
    actual = y_test[i]
    prob_not_spam = y_test_proba[i][0]
    prob_spam = y_test_proba[i][1]
    predicted = y_test_pred[i]
    correct = "✓" if actual == predicted else "✗"
    print(f"{i:5} | {actual:6} | {prob_not_spam:14.3f} | {prob_spam:10.3f} | {predicted:9} | {correct}")

print(f"\nExample Interpretations:")
print(f"  Email 0: {y_test_proba[0][1]*100:.1f}% chance of being spam")
print(f"  Email 1: {y_test_proba[1][1]*100:.1f}% chance of being spam")

# Step 7: Evaluate Model
print("\n" + "="*70)
print("[STEP 7] Evaluating Model Performance...")
print("="*70)

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix (Test Set):")
print(f"                 Predicted Negative  Predicted Positive")
print(f"Actual Negative:  TN={tn:<14} FP={fp}")
print(f"Actual Positive:  FN={fn:<14} TP={tp}")

print(f"\nConfusion Matrix Explanation:")
print(f"  TN (True Negative): {tn} correctly identified as NOT spam")
print(f"  FP (False Positive): {fp} non-spam emails marked as spam")
print(f"  FN (False Negative): {fn} spam emails marked as NOT spam")
print(f"  TP (True Positive): {tp} correctly identified as spam")

# Calculate Metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
mcc = matthews_corrcoef(y_test, y_test_pred)

print(f"\n" + "="*70)
print("PERFORMANCE METRICS")
print("="*70)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  → Percentage of correct predictions")
print(f"  → Formula: (TP + TN) / Total")
print(f"  → {tp} + {tn} / {len(y_test)} = {accuracy:.4f}")

print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  → Of emails predicted as SPAM, how many were actually spam?")
print(f"  → Formula: TP / (TP + FP)")
print(f"  → {tp} / ({tp} + {fp}) = {precision:.4f}")
print(f"  → If precision is low: many non-spam marked as spam (false alarms)")

print(f"\nRecall: {recall:.4f} ({recall*100:.2f}%)")
print(f"  → Of actual SPAM emails, how many did we catch?")
print(f"  → Formula: TP / (TP + FN)")
print(f"  → {tp} / ({tp} + {fn}) = {recall:.4f}")
print(f"  → If recall is low: many spam emails slip through")

print(f"\nF1-Score: {f1:.4f}")
print(f"  → Balance between Precision and Recall")
print(f"  → Formula: 2 * (precision * recall) / (precision + recall)")
print(f"  → Good F1 means both precision and recall are good")

print(f"\nROC-AUC Score: {roc_auc:.4f}")
print(f"  → Probability model ranks random positive higher than negative")
print(f"  → 0.5 = random classifier, 1.0 = perfect classifier")
print(f"  → Values > 0.7 are generally good")

print(f"\nMatthews Correlation Coefficient: {mcc:.4f}")
print(f"  → Correlation between predictions and actual")
print(f"  → More balanced than accuracy for imbalanced data")

# Classification Report
print(f"\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_test_pred, 
                          target_names=['Not Spam', 'Spam'],
                          digits=4))

# Training vs Testing
print(f"\n" + "="*70)
print("TRAINING vs TESTING COMPARISON")
print("="*70)

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f"\nTraining Accuracy: {accuracy_train:.4f}")
print(f"Testing Accuracy: {accuracy_test:.4f}")
print(f"Difference: {abs(accuracy_train - accuracy_test):.4f}")

if abs(accuracy_train - accuracy_test) < 0.05:
    print("✓ No significant overfitting detected!")
else:
    print("⚠ Possible overfitting - consider regularization")

# Step 8: Visualize Results
print("\n" + "="*70)
print("[STEP 8] Visualizing Results...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Plot 1: Confusion Matrix
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
            xticklabels=['Not Spam', 'Spam'],
            yticklabels=['Not Spam', 'Spam'])
ax1.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax1.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax1.set_title('Confusion Matrix - Test Set', fontsize=12, fontweight='bold')

# Add percentages
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm[i].sum() * 100
        ax1.text(j + 0.5, i + 0.7, f'({percentage:.0f}%)',
                ha='center', va='center', color='red', fontweight='bold')

# Plot 2: ROC Curve
ax2 = axes[0, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])
ax2.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier (AUC = 0.5)')
ax2.fill_between(fpr, tpr, alpha=0.2)
ax2.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

# Plot 3: Probability Distribution
ax3 = axes[1, 0]
ax3.hist(y_test_proba[y_test == 0, 1], bins=15, alpha=0.6, label='Not Spam', color='green')
ax3.hist(y_test_proba[y_test == 1, 1], bins=15, alpha=0.6, label='Spam', color='red')
ax3.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
ax3.set_xlabel('Predicted Probability of Spam', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Probability Distribution by Class', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Metrics Comparison
ax4 = axes[1, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [accuracy, precision, recall, f1, roc_auc]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
ax4.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1.1])
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('logistic_regression_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'logistic_regression_analysis.png'")
plt.show()

# Step 9: Threshold Analysis
print("\n" + "="*70)
print("[STEP 9] Threshold Analysis")
print("="*70)

print("\nTesting different decision thresholds:")
print("\nThreshold | Precision | Recall | F1-Score | Interpretation")
print("-" * 70)

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_custom = (y_test_proba[:, 1] >= threshold).astype(int)
    prec = precision_score(y_test, y_pred_custom, zero_division=0)
    rec = recall_score(y_test, y_pred_custom, zero_division=0)
    f1_custom = f1_score(y_test, y_pred_custom, zero_division=0)
    
    if threshold < 0.5:
        interp = "Catch more spam (Higher Recall)"
    elif threshold > 0.5:
        interp = "Fewer false positives (Higher Precision)"
    else:
        interp = "Balanced (Default)"
    
    print(f"{threshold:9.1f} | {prec:9.3f} | {rec:6.3f} | {f1_custom:8.3f} | {interp}")

print("\nKey Insight:")
print("  Lower threshold → Catch more spam but more false alarms")
print("  Higher threshold → Fewer false alarms but miss more spam")
print("  Choose threshold based on business needs!")

# Step 10: Summary
print("\n" + "="*70)
print("[SUMMARY] Key Findings")
print("="*70)

print(f"""
📊 EXPERIMENT SUMMARY:

1. CLASSIFICATION TASK:
   Problem: Detect spam emails
   Classes: Not Spam (0) and Spam (1)
   Total samples: {len(data)}
   Class distribution: {(y==0).sum()} not spam, {(y==1).sum()} spam

2. KEY METRICS:
   Accuracy: {accuracy:.4f} - Overall correctness
   Precision: {precision:.4f} - Of predictions, how many correct
   Recall: {recall:.4f} - Of actual spam, how many caught
   F1-Score: {f1:.4f} - Balance between precision and recall
   ROC-AUC: {roc_auc:.4f} - Model discrimination ability

3. CONFUSION MATRIX:
   True Negatives: {tn} (correctly identified non-spam)
   False Positives: {fp} (non-spam wrongly marked as spam)
   False Negatives: {fn} (spam wrongly marked as non-spam)
   True Positives: {tp} (correctly identified spam)

4. FEATURE IMPORTANCE (by coefficient magnitude):
""")

feature_names = ['word_count', 'link_count', 'caps_ratio']
coefficients = model.coef_[0]
feature_importance = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)

for i, (name, coef) in enumerate(feature_importance, 1):
    direction = "↑" if coef > 0 else "↓"
    print(f"   {i}. {name}: {direction} (coef={coef:.4f})")

print(f"""
5. RECOMMENDATIONS:
""")

if roc_auc > 0.8:
    print("   ✓ Model performs very well! Ready for deployment.")
elif roc_auc > 0.7:
    print("   ✓ Model performs well. Consider optimizing threshold.")
else:
    print("   ⚠ Model needs improvement. Try:")
    print("     - Adding more features")
    print("     - Collecting more data")
    print("     - Feature engineering")

if abs(accuracy_train - accuracy_test) > 0.1:
    print("   ⚠ Possible overfitting. Consider:")
    print("     - Regularization (C parameter)")
    print("     - More training data")
else:
    print("   ✓ No significant overfitting detected.")

print(f"""
6. NEXT STEPS:
   • Optimize decision threshold for business needs
   • Collect more data if available
   • Try ensemble methods (Experiment 3)
   • Deploy with monitoring

✅ Experiment 2 Complete! You now understand Logistic Regression.
""")

print("="*70)
print("Ready for Experiment 3: Decision Trees & Random Forests? 🚀")
print("="*70)
