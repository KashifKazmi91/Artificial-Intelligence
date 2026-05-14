# Experiment 1: Linear Regression - Understanding the Basics

## 🎯 Learning Objectives

After completing this experiment, you will understand:
- What linear regression is and how it works
- How to prepare data for machine learning
- How to train a machine learning model
- How to make predictions
- How to evaluate model performance
- Key concepts: hypothesis, cost function, gradient descent

## 📖 Concept Explanation

### What is Linear Regression?

Linear regression is the simplest form of machine learning. It finds the **best-fitting straight line** through your data points.

**Simple Analogy**: Imagine you want to predict house prices based on their size. Linear regression draws a straight line through your house data that best represents the relationship between size and price.

### The Mathematical Concept

```
y = mx + b
```

Where:
- `y` = predicted value (e.g., house price)
- `m` = slope (how much y changes when x increases by 1)
- `x` = input feature (e.g., house size)
- `b` = intercept (y value when x is 0)

### How Does It Learn?

1. **Start with random values** for m and b
2. **Calculate error** - how far predictions are from actual values
3. **Adjust m and b** to reduce error (using gradient descent)
4. **Repeat** until error is minimal

### Key Concepts Explained

#### Cost Function (Loss)
Measures how wrong our predictions are:
```
MSE = (1/n) * Σ(predicted - actual)²
```
Goal: Make this as small as possible

#### Gradient Descent
An optimization algorithm that:
- Calculates how much to change m and b
- Updates them in the direction that reduces error
- Repeats until convergence (improvement stops)

#### R² Score
Measures how well the model fits the data:
- 1.0 = perfect fit
- 0.5 = okay fit
- 0.0 or negative = poor fit

## 🔧 Step-by-Step Code Explanation

### Step 1: Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
```

What each does:
- **numpy**: Mathematical operations
- **pandas**: Data manipulation
- **sklearn**: Machine learning tools
- **matplotlib**: Visualization

### Step 2: Load and Explore Data
```python
# Load data
data = pd.read_csv('housing_data.csv')

# Explore
print(data.head())      # First 5 rows
print(data.info())      # Data types and missing values
print(data.describe())  # Statistics
```

Why explore?
- Understand what you're working with
- Detect missing or invalid values
- Check if data makes sense

### Step 3: Prepare Data
```python
# Separate features (X) and target (y)
X = data[['square_feet']]  # Input features
y = data['price']          # Target to predict

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Why split?
- **Training set**: Learn patterns
- **Testing set**: Evaluate performance on unseen data
- Prevents overfitting (memorizing the data)

### Step 4: Train the Model
```python
# Create model
model = LinearRegression()

# Train on training data
model.fit(X_train, y_train)
```

What happens inside:
1. Model finds best m and b
2. Minimizes prediction errors
3. Learns the relationship between features and target

### Step 5: Make Predictions
```python
# Predict on test set
y_pred = model.predict(X_test)

# Predict on new unseen data
new_house_size = [[3000]]  # 3000 sq ft
predicted_price = model.predict(new_house_size)
print(f"Predicted price: ${predicted_price[0]:,.2f}")
```

### Step 6: Evaluate Performance
```python
# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Print model parameters
print(f"\nModel Equation: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
print(f"For every 1 sq ft increase, price increases by ${model.coef_[0]:.2f}")
```

### Step 7: Visualize Results
```python
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression: House Price Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 💻 Running the Code

### Option 1: Run Python Script
```bash
python code.py
```

### Option 2: Run Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```

## 📊 What You Should See

1. **Model Evaluation Metrics**
   - RMSE tells you average prediction error in dollars
   - R² score tells you how well the model explains price variations

2. **Model Equation**
   - Shows the relationship: Price = (coefficient × size) + intercept
   - Coefficient interpretation: how much price changes per unit feature

3. **Visualization**
   - Blue dots: actual prices
   - Red line: predicted prices
   - How close they are = how good the model is

## 🧠 Common Issues & How to Fix Them

### Issue 1: Model Performance is Poor (Low R²)
**Causes:**
- Relationship isn't actually linear
- Important features are missing
- Data has outliers

**Solutions:**
- Try polynomial regression (curved line)
- Add more relevant features
- Remove extreme outliers

### Issue 2: Predictions are Always the Same
**Cause:** Feature scaling issue or no variance in data

**Solution:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

### Issue 3: Negative R² Score
**Cause:** Model is worse than just predicting the average

**Solution:**
- Feature selection: choose better predictive features
- Check data quality: look for errors in data
- Increase training data

## 🎓 Key Takeaways

✅ Linear regression finds the best-fitting straight line
✅ It learns by minimizing prediction error
✅ Train/test split prevents overfitting
✅ RMSE and R² tell you how good the model is
✅ Model coefficients show feature importance
✅ Always visualize to verify results

## 🚀 Next Steps

1. **Experiment with the code**:
   - Change the test_size to 0.3 (30% test data)
   - Try different random_state values
   - Plot residuals (prediction errors)

2. **Try different datasets**:
   - [Boston Housing Dataset](https://www.kaggle.com/c/boston-housing)
   - [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
   - [Your own data](https://www.kaggle.com)

3. **Advanced topics**:
   - Multiple linear regression (multiple features)
   - Feature scaling and normalization
   - Regularization (L1, L2)
   - Polynomial regression

## 📚 Resources

- [Scikit-learn Linear Regression Docs](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [3Blue1Brown: Essence of Linear Regression](https://www.youtube.com/watch?v=PXAub6oion8) (Video)
- [Andrew Ng's ML Course - Linear Regression](https://www.coursera.org/learn/machine-learning)

## ❓ Practice Questions

1. What would happen if you used all the data for training and none for testing?
2. Why do we use RMSE instead of just MSE?
3. What does a coefficient of 5 mean in the equation y = 5x + 10?
4. How would the model change if we added an outlier (very expensive house)?
5. Can linear regression predict non-linear relationships perfectly?

## 📋 Checklist Before Moving to Experiment 2

- [ ] Understand what linear regression does
- [ ] Can explain cost function and gradient descent
- [ ] Can run the code and understand each step
- [ ] Understand train/test split and why it matters
- [ ] Know how to interpret R² and RMSE
- [ ] Can create and interpret visualizations
- [ ] Completed practice questions

Once you've checked all boxes, you're ready for **Experiment 2: Logistic Regression**! 🎉
