# Experiment 3: Decision Trees & Random Forests - Ensemble Learning

## 🎯 Learning Objectives

After completing this experiment, you will understand:
- How decision trees make decisions
- Tree splitting and information gain
- Overfitting in trees and how to control it
- Ensemble methods and why they work better
- Random forests and their advantages
- Feature importance from tree-based models

## 📖 Concept Explanation

### What is a Decision Tree?

A decision tree mimics human decision-making:
- Starts at the root (top)
- Asks a series of YES/NO questions
- Each question splits the data
- Reaches a leaf (final decision)

**Simple Example**: Classifying iris flowers
```
Start: All flowers
├─ Is petal length > 2.5cm?
│  ├─ YES → Is petal width > 1.7cm?
│  │  ├─ YES → Virginica
│  │  └─ NO → Versicolor
│  └─ NO → Setosa
```

### Key Concepts

#### Gini Impurity
Measures how mixed the data is (0 = pure, 1 = mixed)
- Lower Gini = better split
- Used to decide which feature to split on

#### Information Gain
Reduction in impurity after a split
- Higher gain = better split
- Trees grow by maximizing information gain

#### Entropy
Another measure of uncertainty
- High entropy = uncertain
- Low entropy = certain

### Why Random Forests?

Decision Trees have a problem: **Overfitting**
- A single tree can memorize the training data
- Performs poorly on new data

**Solution: Random Forests**
- Train multiple random trees
- Each tree learns different patterns
- Average predictions (voting)
- Reduces overfitting significantly

**Key Features of Random Forests**:
1. **Bootstrap Sampling**: Each tree gets random subset of data
2. **Random Features**: Each split considers random subset of features
3. **Voting/Averaging**: Final prediction is majority vote or average
4. **Out-of-Bag Score**: Automatic validation using unused data

## 🔧 Step-by-Step Code Explanation

### Step 1: Import Libraries
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
```

### Step 2: Prepare Data
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Step 3: Train Single Decision Tree
```python
# Without depth limit (prone to overfitting)
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

# With depth limit (prevents overfitting)
dt_limited = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_limited.fit(X_train, y_train)
```

**Key Parameters**:
- `max_depth`: Maximum levels deep the tree can grow
- `min_samples_split`: Minimum samples needed to split
- `min_samples_leaf`: Minimum samples in leaf node

### Step 4: Train Random Forest
```python
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Max depth per tree
    min_samples_split=5,   # Min samples to split
    n_jobs=-1,             # Use all cores
    random_state=42
)
rf.fit(X_train, y_train)
```

### Step 5: Make Predictions
```python
# Single tree prediction
y_pred_dt = dt.predict(X_test)

# Random forest prediction
y_pred_rf = rf.predict(X_test)

# Get probabilities
y_proba_rf = rf.predict_proba(X_test)
```

### Step 6: Feature Importance
```python
importances = rf.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")
```

Shows which features matter most for predictions

### Step 7: Visualize Tree
```python
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_limited, 
          feature_names=feature_names,
          class_names=['Class0', 'Class1'],
          ax=ax,
          filled=True)
plt.show()
```

## 📊 What You Should See

### 1. Tree Structure
- Each node shows the splitting condition
- Colors show class dominance
- Depth indicates tree complexity

### 2. Performance Comparison
```
Decision Tree (Full): Accuracy = 0.95 (training), 0.85 (testing) → Overfitting
Decision Tree (Limited): Accuracy = 0.92 (training), 0.90 (testing) → Better
Random Forest: Accuracy = 0.93 (training), 0.92 (testing) → Best
```

### 3. Feature Importance
```
Feature 1: 0.45 (Most important)
Feature 2: 0.30
Feature 3: 0.20
Feature 4: 0.05
```

## 🧠 Common Issues & Solutions

### Issue 1: Decision Tree Overfitting
**Problem**: Perfect training accuracy, poor testing accuracy

**Solution**:
```python
dt = DecisionTreeClassifier(
    max_depth=5,              # Limit depth
    min_samples_split=10,     # More samples to split
    min_samples_leaf=5        # More samples per leaf
)
```

### Issue 2: Random Forest Slow
**Problem**: Takes too long to train

**Solution**:
```python
rf = RandomForestClassifier(
    n_estimators=50,  # Reduce trees
    n_jobs=-1,        # Parallel processing
    max_depth=10      # Limit depth
)
```

### Issue 3: Feature Importance All Same
**Problem**: Can't identify important features

**Cause**: Features have different scales

**Solution**: Scale features before training
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 💡 Key Takeaways

✅ Decision trees use hierarchical YES/NO questions
✅ Easy to visualize and interpret
✅ Prone to overfitting without depth limits
✅ Random forests combine many trees to reduce overfitting
✅ Feature importance shows which variables matter
✅ Ensemble methods usually outperform single models
✅ Out-of-bag scores provide free validation

## 🚀 Next Steps

1. **Experiment with hyperparameters**:
   - Change `max_depth`: 3, 5, 10, 15, 20
   - Change `n_estimators`: 10, 50, 100, 200
   - Observe how accuracy changes

2. **Try different datasets**:
   - [Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
   - [Wine Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
   - [Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

3. **Advanced topics**:
   - Gradient Boosting Machines (XGBoost, LightGBM)
   - AdaBoost ensemble
   - Permutation importance
   - SHAP values for explanations

## 📚 Resources

- [Scikit-learn Tree Docs](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forest Guide](https://en.wikipedia.org/wiki/Random_forest)
- [Information Gain Explained](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)

## ✓ Checklist Before Moving to Experiment 4

- [ ] Understand how decision trees make splits
- [ ] Know what Gini impurity and information gain are
- [ ] Understand why single trees overfit
- [ ] Know how random forests reduce overfitting
- [ ] Can interpret feature importance
- [ ] Can tune hyperparameters
- [ ] Can visualize and explain a decision tree
- [ ] Completed practice questions

Once you've checked all boxes, you're ready for **Experiment 4: Neural Networks**! 🎉
