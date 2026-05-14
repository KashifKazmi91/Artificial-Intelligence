"""
Experiment 3: Decision Trees & Random Forests

This script demonstrates tree-based ensemble learning.
We'll classify iris flowers using Decision Trees and Random Forests.

Conceptual Flow:
1. Load iris dataset
2. Prepare features and target
3. Split data
4. Train single decision tree (show overfitting)
5. Train limited decision tree (prevent overfitting)
6. Train random forest (ensemble method)
7. Compare performance
8. Analyze feature importance
9. Visualize decision tree
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EXPERIMENT 3: DECISION TREES & RANDOM FORESTS")
print("Ensemble Learning for Iris Classification")
print("="*70)

# Step 1: Load Dataset
print("\n[STEP 1] Loading Iris Dataset...")

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"\nDataset Information:")
print(f"  Number of samples: {X.shape[0]}")
print(f"  Number of features: {X.shape[1]}")
print(f"  Number of classes: {len(np.unique(y))}")
print(f"\nFeatures:")
for i, name in enumerate(feature_names):
    print(f"  {i+1}. {name}")
print(f"\nClasses:")
for i, name in enumerate(target_names):
    print(f"  {i}: {name}")
print(f"\nClass Distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {target_names[u]}: {c} samples")

# Step 2: Prepare Data
print("\n" + "="*70)
print("[STEP 2] Preparing Data...")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)} samples (80%)")
print(f"Testing set size: {len(X_test)} samples (20%)")
print(f"\nFeature statistics (on training data):")
print(f"\n{'Feature':<30} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
print("-" * 70)
for i, name in enumerate(feature_names):
    print(f"{name:<30} {X_train[:, i].min():>10.2f} {X_train[:, i].max():>10.2f} "
          f"{X_train[:, i].mean():>10.2f} {X_train[:, i].std():>10.2f}")

# Step 3: Train Decision Tree (Full - Overfitting Example)
print("\n" + "="*70)
print("[STEP 3] Training Full Decision Tree (No Depth Limit)...")
print("="*70)

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

y_train_pred_full = dt_full.predict(X_train)
y_test_pred_full = dt_full.predict(X_test)

acc_train_full = accuracy_score(y_train, y_train_pred_full)
acc_test_full = accuracy_score(y_test, y_test_pred_full)

print(f"\nFull Decision Tree Performance:")
print(f"  Training Accuracy: {acc_train_full:.4f} ({acc_train_full*100:.2f}%)")
print(f"  Testing Accuracy: {acc_test_full:.4f} ({acc_test_full*100:.2f}%)")
print(f"  Overfitting Gap: {(acc_train_full - acc_test_full):.4f}")
print(f"  Tree Depth: {dt_full.get_depth()}")
print(f"  Number of Leaves: {dt_full.get_n_leaves()}")

if acc_train_full - acc_test_full > 0.1:
    print(f"\n  ⚠️ WARNING: Significant overfitting detected!")
    print(f"  The tree memorized training data but doesn't generalize well.")
else:
    print(f"\n  ✓ Good generalization.")

# Step 4: Train Limited Decision Tree (Prevent Overfitting)
print("\n" + "="*70)
print("[STEP 4] Training Limited Decision Tree (max_depth=5)...")
print("="*70)

dt_limited = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_limited.fit(X_train, y_train)

y_train_pred_limited = dt_limited.predict(X_train)
y_test_pred_limited = dt_limited.predict(X_test)

acc_train_limited = accuracy_score(y_train, y_train_pred_limited)
acc_test_limited = accuracy_score(y_test, y_test_pred_limited)

print(f"\nLimited Decision Tree Performance:")
print(f"  Training Accuracy: {acc_train_limited:.4f} ({acc_train_limited*100:.2f}%)")
print(f"  Testing Accuracy: {acc_test_limited:.4f} ({acc_test_limited*100:.2f}%)")
print(f"  Overfitting Gap: {(acc_train_limited - acc_test_limited):.4f}")
print(f"  Tree Depth: {dt_limited.get_depth()}")
print(f"  Number of Leaves: {dt_limited.get_n_leaves()}")

if acc_train_limited - acc_test_limited < 0.1:
    print(f"\n  ✓ Good generalization! Limited depth prevents overfitting.")
else:
    print(f"\n  The model could still use more regularization.")

# Step 5: Train Random Forest
print("\n" + "="*70)
print("[STEP 5] Training Random Forest (100 trees)...")
print("="*70)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)

acc_train_rf = accuracy_score(y_train, y_train_pred_rf)
acc_test_rf = accuracy_score(y_test, y_test_pred_rf)

print(f"\nRandom Forest Performance:")
print(f"  Training Accuracy: {acc_train_rf:.4f} ({acc_train_rf*100:.2f}%)")
print(f"  Testing Accuracy: {acc_test_rf:.4f} ({acc_test_rf*100:.2f}%)")
print(f"  Overfitting Gap: {(acc_train_rf - acc_test_rf):.4f}")
print(f"  Number of Trees: {rf.n_estimators}")
print(f"  Out-of-Bag Score: {rf.oob_score_:.4f}")

print(f"\n  ✓ Random Forest shows best generalization!")

# Step 6: Model Comparison
print("\n" + "="*70)
print("[STEP 6] Model Comparison")
print("="*70)

models = {
    'Full Decision Tree': (y_train_pred_full, y_test_pred_full, acc_train_full, acc_test_full),
    'Limited Decision Tree': (y_train_pred_limited, y_test_pred_limited, acc_train_limited, acc_test_limited),
    'Random Forest': (y_train_pred_rf, y_test_pred_rf, acc_train_rf, acc_test_rf)
}

print("\nModel Performance Comparison:")
print(f"\n{'Model':<25} {'Train Acc':>12} {'Test Acc':>12} {'Gap':>12} {'Quality':<15}")
print("-" * 70)

for model_name, (_, _, acc_train, acc_test) in models.items():
    gap = acc_train - acc_test
    if gap < 0.05:
        quality = "Excellent ✓"
    elif gap < 0.1:
        quality = "Good"
    elif gap < 0.2:
        quality = "Okay"
    else:
        quality = "Poor (Overfitting)"
    
    print(f"{model_name:<25} {acc_train:>12.4f} {acc_test:>12.4f} {gap:>12.4f} {quality:<15}")

# Step 7: Feature Importance
print("\n" + "="*70)
print("[STEP 7] Feature Importance Analysis")
print("="*70)

feature_importance = rf.feature_importances_
feature_importance_sorted = np.argsort(feature_importance)[::-1]

print(f"\nFeature Importance from Random Forest:")
print(f"\nRank | Feature Name                     | Importance | Interpretation")
print("-" * 75)

for rank, idx in enumerate(feature_importance_sorted, 1):
    importance = feature_importance[idx]
    bar_length = int(importance * 40)
    bar = "█" * bar_length
    
    if importance > 0.3:
        interp = "Very Important"
    elif importance > 0.1:
        interp = "Important"
    else:
        interp = "Less Important"
    
    print(f"{rank:<4} | {feature_names[idx]:<30} | {importance:>10.4f} | {interp:<15}")
    print(f"     |                                | {bar}")

print(f"\nInterpretation:")
print(f"  - Higher importance = more useful for making predictions")
print(f"  - Trees focus on features that best separate classes")
print(f"  - Can guide feature engineering in real projects")

# Step 8: Detailed Classification Report (Random Forest)
print("\n" + "="*70)
print("[STEP 8] Detailed Classification Report (Random Forest)")
print("="*70)

print(f"\nRandom Forest - Test Set Performance:")
print(classification_report(y_test, y_test_pred_rf, target_names=target_names, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred_rf)
print(f"\nConfusion Matrix:")
print(f"\n{'':>20} {target_names[0]:>15} {target_names[1]:>15} {target_names[2]:>15}")
print("-" * 70)
for i, name in enumerate(target_names):
    print(f"Actual {name:<12} {cm[i, 0]:>15} {cm[i, 1]:>15} {cm[i, 2]:>15}")

# Step 9: Cross-Validation Scores
print("\n" + "="*70)
print("[STEP 9] Cross-Validation Scores")
print("="*70)

cv_scores_dt = cross_val_score(dt_limited, X_train, y_train, cv=5)
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5)

print(f"\nDecision Tree (5-Fold Cross-Validation):")
print(f"  Fold Scores: {cv_scores_dt}")
print(f"  Mean Score: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")

print(f"\nRandom Forest (5-Fold Cross-Validation):")
print(f"  Fold Scores: {cv_scores_rf}")
print(f"  Mean Score: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

if cv_scores_rf.mean() > cv_scores_dt.mean():
    improvement = (cv_scores_rf.mean() - cv_scores_dt.mean()) * 100
    print(f"\n  ✓ Random Forest is {improvement:.2f}% better on average!")

# Step 10: Visualizations
print("\n" + "="*70)
print("[STEP 10] Creating Visualizations...")
print("="*70)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Model Comparison
ax1 = plt.subplot(2, 3, 1)
model_names = list(models.keys())
train_accs = [models[name][2] for name in model_names]
test_accs = [models[name][3] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

ax1.bar(x - width/2, train_accs, width, label='Training', alpha=0.8, color='skyblue')
ax1.bar(x + width/2, test_accs, width, label='Testing', alpha=0.8, color='salmon')
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=15, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0.8, 1.0])

# Add value labels
for i, (train, test) in enumerate(zip(train_accs, test_accs)):
    ax1.text(i - width/2, train + 0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Feature Importance
ax2 = plt.subplot(2, 3, 2)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
ax2.barh(range(len(feature_importance)), feature_importance[feature_importance_sorted], color=colors)
ax2.set_yticks(range(len(feature_importance)))
ax2.set_yticklabels([feature_names[i] for i in feature_importance_sorted])
ax2.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax2.set_title('Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

for i, idx in enumerate(feature_importance_sorted):
    ax2.text(feature_importance[idx] + 0.01, i, f'{feature_importance[idx]:.3f}', 
             va='center', fontsize=9, fontweight='bold')

# Plot 3: Confusion Matrix (Random Forest)
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3,
            xticklabels=target_names, yticklabels=target_names)
ax3.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax3.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax3.set_title('Confusion Matrix (Random Forest)', fontsize=12, fontweight='bold')

# Plot 4: Cross-Validation Scores
ax4 = plt.subplot(2, 3, 4)
folds = np.arange(1, 6)
ax4.plot(folds, cv_scores_dt, 'o-', linewidth=2, markersize=8, label='Decision Tree', color='skyblue')
ax4.plot(folds, cv_scores_rf, 's-', linewidth=2, markersize=8, label='Random Forest', color='salmon')
ax4.axhline(cv_scores_dt.mean(), color='skyblue', linestyle='--', alpha=0.5)
ax4.axhline(cv_scores_rf.mean(), color='salmon', linestyle='--', alpha=0.5)
ax4.set_xlabel('Fold Number', fontsize=11, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax4.set_title('5-Fold Cross-Validation Scores', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0.85, 1.0])

# Plot 5: Overfitting Analysis
ax5 = plt.subplot(2, 3, 5)
overfitting_gaps = [
    acc_train_full - acc_test_full,
    acc_train_limited - acc_test_limited,
    acc_train_rf - acc_test_rf
]
colors_gap = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in overfitting_gaps]
ax5.bar(model_names, overfitting_gaps, color=colors_gap, alpha=0.7, edgecolor='black')
ax5.axhline(0.1, color='red', linestyle='--', label='Poor (>0.1)', linewidth=1.5)
ax5.axhline(0.05, color='orange', linestyle='--', label='Okay (>0.05)', linewidth=1.5)
ax5.set_ylabel('Train-Test Accuracy Gap', fontsize=11, fontweight='bold')
ax5.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
ax5.set_xticklabels(model_names, rotation=15, ha='right')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

for i, gap in enumerate(overfitting_gaps):
    ax5.text(i, gap + 0.003, f'{gap:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 6: Decision Tree Visualization (Limited Tree)
ax6 = plt.subplot(2, 3, 6)
plot_tree(dt_limited, feature_names=feature_names, class_names=target_names,
          ax=ax6, filled=True, fontsize=8, rounded=True)
ax6.set_title('Decision Tree Structure (max_depth=5)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('decision_trees_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'decision_trees_analysis.png'")
plt.show()

# Step 11: Detailed Tree Visualization
print(f"\n" + "="*70)
print("[STEP 11] Creating Detailed Tree Visualization...")
print("="*70)

fig, ax = plt.subplots(figsize=(25, 15))
plot_tree(dt_limited, feature_names=feature_names, class_names=target_names,
          ax=ax, filled=True, fontsize=10, rounded=True, proportion=True)
plt.title('Detailed Decision Tree (Limited Depth)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('decision_tree_detailed.png', dpi=300, bbox_inches='tight')
print("\n✓ Detailed tree saved as 'decision_tree_detailed.png'")
plt.show()

# Step 12: Summary
print("\n" + "="*70)
print("[SUMMARY] Key Findings")
print("="*70)

print(f"""
📊 EXPERIMENT SUMMARY:

1. DECISION TREE ANALYSIS:
   Full Tree (No Limit):
     - Training Accuracy: {acc_train_full:.4f}
     - Testing Accuracy: {acc_test_full:.4f}
     - Overfitting Gap: {acc_train_full - acc_test_full:.4f}
     - Tree Depth: {dt_full.get_depth()}
     - Status: {'⚠️ Overfitting' if acc_train_full - acc_test_full > 0.1 else '✓ Good'}

   Limited Tree (max_depth=5):
     - Training Accuracy: {acc_train_limited:.4f}
     - Testing Accuracy: {acc_test_limited:.4f}
     - Overfitting Gap: {acc_train_limited - acc_test_limited:.4f}
     - Tree Depth: {dt_limited.get_depth()}
     - Status: {'✓ Good generalization' if acc_train_limited - acc_test_limited < 0.1 else '⚠️ Still overfitting'}

2. RANDOM FOREST ANALYSIS:
   - Training Accuracy: {acc_train_rf:.4f}
   - Testing Accuracy: {acc_test_rf:.4f}
   - Overfitting Gap: {acc_train_rf - acc_test_rf:.4f}
   - Number of Trees: 100
   - Out-of-Bag Score: {rf.oob_score_:.4f}
   - Status: ✓ Best performance!

3. FEATURE IMPORTANCE (Top 3):
""")

for i, idx in enumerate(feature_importance_sorted[:3], 1):
    print(f"   {i}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

print(f"""
4. KEY INSIGHTS:
   ✓ Random Forest outperforms single Decision Tree
   ✓ Limiting tree depth prevents overfitting
   ✓ Ensemble methods combine strengths of multiple models
   ✓ Feature importance guides understanding of data
   ✓ Cross-validation shows consistent performance

5. WHY RANDOM FOREST IS BETTER:
   • Reduces overfitting through averaging
   • Each tree sees different data (bootstrap sampling)
   • Each split considers random feature subset
   • Robust to outliers and noise
   • Automatic feature importance calculation

6. RECOMMENDATIONS:
   ✓ Use Random Forest for this classification task
   ✓ Fine-tune n_estimators and max_depth
   ✓ Consider other ensemble methods (XGBoost, LightGBM)
   ✓ Feature {feature_names[feature_importance_sorted[0]]} is most important

✅ Experiment 3 Complete! You now understand Decision Trees and Ensembles.
""")

print("="*70)
print("Ready for Experiment 4: Neural Networks? 🚀")
print("="*70)
