# Experiment 2: Logistic Regression - Classification Problems

## 🎯 Learning Objectives

After completing this experiment, you will understand:
- What classification is and how it differs from regression
- How logistic regression works with probabilities
- Binary classification problems
- Decision boundaries and thresholds
- How to handle imbalanced datasets
- Key concepts: sigmoid function, decision boundary, probability threshold

## 📖 Concept Explanation

### What is Classification?

**Classification** answers YES/NO questions:
- Is this email spam? (Yes/No)
- Will this customer buy? (Yes/No)
- Is this tumor malignant? (Yes/No)
- Will this loan default? (Yes/No)

Unlike regression that predicts continuous values, classification predicts categories.

### What is Logistic Regression?

Despite its name, **logistic regression is a classification algorithm**, not regression.

It:
1. Calculates a probability (0 to 1)
2. Uses the probability to classify into categories
3. Draws a decision boundary between classes

### The Sigmoid Function

Logistic regression uses the **sigmoid function**:
```
f(x) = 1 / (1 + e^(-z))
```

Where:
- Output is always between 0 and 1 (perfect for probability)
- 0.5 is the threshold (>0.5 = Class 1, ≤0.5 = Class 0)
- Creates an S-shaped curve

### Simple Analogy

Imagine you're a bank deciding who gets a loan:
- Features: Income, credit score, employment history
- Output: Probability of repaying (0.0 to 1.0)
- Decision: If probability > 0.7, approve the loan

### Key Concepts Explained

#### Binary Classification
- Class 0 (Negative): No, Not approved, Negative
- Class 1 (Positive): Yes, Approved, Positive

#### Decision Boundary
The line (or hyperplane) that separates the two classes.
- Points above the line → Class 1
- Points below the line → Class 0

#### Probability Threshold
The cutoff probability (usually 0.5):
- If P(Class 1) > 0.5 → Predict Class 1
- If P(Class 1) ≤ 0.5 → Predict Class 0

#### Confusion Matrix
```
                Predicted Positive    Predicted Negative
Actually Positive:  True Positive      False Negative
Actually Negative:  False Positive     True Negative
```

#### Important Metrics
- **Accuracy**: (TP + TN) / Total — Overall correctness
- **Precision**: TP / (TP + FP) — Of predictions, how many correct?
- **Recall**: TP / (TP + FN) — Of actual positives, how many found?
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: How well model distinguishes classes

## 🔧 Step-by-Step Code Explanation

### Step 1: Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
```

New additions for classification:
- **confusion_matrix**: See classification breakdown
- **accuracy, precision, recall**: Different evaluation angles
- **roc_curve, roc_auc_score**: Advanced evaluation

### Step 2: Load and Explore Data
```python
# Load data
data = pd.read_csv('email_spam.csv')

# Check class distribution
print(data['is_spam'].value_counts())
print(data['is_spam'].value_counts(normalize=True))
```

Why check distribution?
- Imbalanced data (e.g., 95% not spam, 5% spam) needs special handling
- Affects which metrics matter most

### Step 3: Prepare Data
```python
# Features (email characteristics)
X = data[['word_count', 'has_links', 'has_attachments']]

# Target (is it spam?)
y = data['is_spam']  # 0 = Not spam, 1 = Spam

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Note the `stratify=y`:
- Keeps class distribution same in train and test
- Important when classes are imbalanced

### Step 4: Train the Model
```python
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
```

What happens inside:
1. Finds the best decision boundary
2. Learns weights for each feature
3. Minimizes classification error

### Step 5: Make Predictions
```python
# Probability predictions (values between 0 and 1)
y_pred_proba = model.predict_proba(X_test)
# Returns array with [P(class 0), P(class 1)] for each sample

# Class predictions (0 or 1)
y_pred = model.predict(X_test)
```

Two types of predictions:
- **Probability**: "This email has 82% chance of being spam"
- **Class**: "This email is spam" (0 or 1)

### Step 6: Evaluate the Model
```python
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy: How many did we get right overall?
accuracy = accuracy_score(y_test, y_pred)

# Precision: Of emails we said were spam, how many were actually spam?
precision = precision_score(y_test, y_pred)

# Recall: Of actual spam emails, how many did we catch?
recall = recall_score(y_test, y_pred)

# F1-Score: Balance between precision and recall
f1 = f1_score(y_test, y_pred)

# ROC-AUC: How well does it distinguish between classes?
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
```

### Step 7: Visualize Results
```python
# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

## 🖥️ Running the Code

### Option 1: Run Python Script
```bash
python code.py
```

### Option 2: Run Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```

## 📊 What You Should See

### 1. Confusion Matrix Output
```
Confusion Matrix:
[[TN  FP]
 [FN  TP]]
```

Example:
```
[[80  5]   <- 80 correct non-spam, 5 spam predicted as non-spam
 [3  12]]  <- 3 non-spam predicted as spam, 12 correct spam
```

### 2. Classification Metrics
```
Accuracy:  0.92    (92% of predictions correct)
Precision: 0.71    (71% of spam predictions actually spam)
Recall:    0.80    (80% of actual spam caught)
F1-Score:  0.75    (Balance between precision and recall)
ROC-AUC:   0.88    (0.5 = random, 1.0 = perfect)
```

### 3. ROC Curve
- Curve above diagonal line = Good model
- Closer to top-left corner = Better performance
- Area under curve (AUC) = Overall performance metric

## 🧠 Common Issues & Solutions

### Issue 1: Imbalanced Classes
**Problem**: Dataset has 95% class 0, 5% class 1

**Solution**:
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

model = LogisticRegression(class_weight='balanced', random_state=42)
```

### Issue 2: Precision vs Recall Trade-off
**Problem**: High precision but low recall (or vice versa)

**Solution**: Adjust decision threshold
```python
# Instead of default 0.5 threshold, use 0.3
y_pred_custom = (y_pred_proba[:, 1] >= 0.3).astype(int)
```

### Issue 3: Features Have Different Scales
**Problem**: Some features 0-1, others 0-1000000

**Solution**: Scale features
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 💡 Key Takeaways

✅ Logistic regression is for binary classification (Yes/No)
✅ Uses sigmoid function to output probabilities
✅ Decision boundary separates the two classes
✅ Multiple evaluation metrics (accuracy, precision, recall, F1, AUC)
✅ Different metrics matter for different problems
✅ Always check confusion matrix for detailed understanding
✅ ROC curve shows trade-off between true and false positives

## 🚀 Next Steps

1. **Experiment with threshold**:
   - Change decision threshold from 0.5 to 0.3, 0.7
   - Observe how precision/recall change

2. **Try different datasets**:
   - [Titanic Dataset](https://www.kaggle.com/c/titanic)
   - [Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
   - [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

3. **Advanced topics**:
   - Multi-class classification (3+ classes)
   - ROC curve optimization
   - Class weight balancing
   - Threshold optimization for business goals

## 📚 Resources

- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Understanding ROC Curves](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Confusion Matrix Explained](https://en.wikipedia.org/wiki/Confusion_matrix)

## ❓ Practice Questions

1. What's the difference between precision and recall? When would you care more about each?
2. Why is accuracy not always the best metric for imbalanced datasets?
3. What does ROC-AUC score of 0.5 mean?
4. How would changing the decision threshold from 0.5 to 0.3 affect precision and recall?
5. In the email spam example, would you prefer high precision or high recall? Why?

## ✅ Checklist Before Moving to Experiment 3

- [ ] Understand binary classification and logistic regression
- [ ] Can explain sigmoid function and decision boundary
- [ ] Know the difference between probability and class predictions
- [ ] Understand confusion matrix and its components
- [ ] Know when to use accuracy vs precision vs recall
- [ ] Can interpret ROC curve and AUC score
- [ ] Can run the code and analyze outputs
- [ ] Completed practice questions

Once you've checked all boxes, you're ready for **Experiment 3: Decision Trees & Random Forests**! 🎉
