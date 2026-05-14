"""
Experiment 1: Linear Regression

This script demonstrates linear regression using Python.
We'll predict house prices based on their size.

Conceptual Flow:
1. Load and explore data
2. Prepare features (X) and target (y)
3. Split into training and testing sets
4. Train the model
5. Make predictions
6. Evaluate performance
7. Visualize results
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EXPERIMENT 1: LINEAR REGRESSION")
print("Predicting House Prices Based on Size")
print("="*70)

# Step 1: Create Sample Dataset
# In real projects, you'd load this from a CSV or database
print("\n[STEP 1] Creating Sample Dataset...")

np.random.seed(42)  # For reproducibility

# House sizes in square feet (features)
house_sizes = np.array([
    1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800,
    3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800
])

# House prices in dollars (target)
# Price ≈ 150 * size + 50000 (with some random variation)
prices = 150 * house_sizes + 50000 + np.random.normal(0, 50000, len(house_sizes))

# Create DataFrame
data = pd.DataFrame({
    'square_feet': house_sizes,
    'price': prices
})

print("\nDataset Preview:")
print(data.head(10))
print(f"\nDataset Shape: {data.shape}")
print(f"\nDataset Statistics:")
print(data.describe())

# Step 2: Prepare Data
print("\n" + "="*70)
print("[STEP 2] Preparing Data...")
print("="*70)

# X = Features (what we use to predict)
X = data[['square_feet']].values  # Must be 2D array

# y = Target (what we want to predict)
y = data['price'].values

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFirst 5 features (X): {X.flatten()[:5]}")
print(f"First 5 targets (y): {y[:5]}")

# Step 3: Train-Test Split
print("\n" + "="*70)
print("[STEP 3] Splitting Data into Training and Testing Sets...")
print("="*70)

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)} samples (80%)")
print(f"Testing set size: {len(X_test)} samples (20%)")
print(f"\nTraining features shape: {X_train.shape}")
print(f"Training targets shape: {y_train.shape}")
print(f"\nTesting features shape: {X_test.shape}")
print(f"Testing targets shape: {y_test.shape}")

# Step 4: Create and Train the Model
print("\n" + "="*70)
print("[STEP 4] Creating and Training the Model...")
print("="*70)

# Create linear regression model
model = LinearRegression()

# Train the model on training data
print("\nTraining the model... (finding best-fit line)")
model.fit(X_train, y_train)
print("✓ Model trained successfully!")

# Model Parameters
print(f"\nModel Parameters:")
print(f"  Slope (coefficient): {model.coef_[0]:.2f}")
print(f"  Intercept: {model.intercept_:.2f}")
print(f"\nModel Equation: price = {model.coef_[0]:.2f} × square_feet + {model.intercept_:.2f}")
print(f"\nInterpretation: For every 1 square foot increase,")
print(f"the house price increases by approximately ${model.coef_[0]:.2f}")

# Step 5: Make Predictions
print("\n" + "="*70)
print("[STEP 5] Making Predictions...")
print("="*70)

# Predictions on training set
y_train_pred = model.predict(X_train)

# Predictions on test set
y_test_pred = model.predict(X_test)

print(f"\nPredictions on Test Set:")
print("\nActual vs Predicted Prices:")
for i in range(min(5, len(X_test))):
    actual = y_test[i]
    predicted = y_test_pred[i]
    error = actual - predicted
    percent_error = (error / actual) * 100
    print(f"  Size: {X_test[i][0]:.0f} sq ft | Actual: ${actual:,.0f} | "
          f"Predicted: ${predicted:,.0f} | Error: ${error:,.0f} ({percent_error:.1f}%)")

# Predict for a new house (not in dataset)
print(f"\nPrediction for a house we haven't seen:")
new_house_sizes = np.array([[2500], [3500], [1500]])  # 3 new houses
new_predictions = model.predict(new_house_sizes)

for size, pred in zip(new_house_sizes, new_predictions):
    print(f"  {size[0]:.0f} sq ft house → Predicted price: ${pred:,.0f}")

# Step 6: Evaluate the Model
print("\n" + "="*70)
print("[STEP 6] Evaluating Model Performance...")
print("="*70)

# Calculate evaluation metrics on TEST set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"\nTest Set Performance Metrics:")
print(f"  Mean Squared Error (MSE): ${mse_test:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse_test:,.2f}")
print(f"  Mean Absolute Error (MAE): ${mae_test:,.2f}")
print(f"  R² Score: {r2_test:.4f}")

print(f"\nMetric Interpretation:")
print(f"  • RMSE ${rmse_test:,.2f}: Average prediction error")
print(f"  • R² Score {r2_test:.4f}: Model explains {r2_test*100:.1f}% of price variation")

if r2_test > 0.8:
    print(f"  ✓ Excellent fit! The model explains price very well.")
elif r2_test > 0.6:
    print(f"  ✓ Good fit. The model explains price reasonably well.")
elif r2_test > 0.3:
    print(f"  ⚠ Moderate fit. Consider adding more features.")
else:
    print(f"  ✗ Poor fit. The linear model may not be appropriate.")

# Calculate for training set (to check for overfitting)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

print(f"\nTraining Set Performance Metrics:")
print(f"  RMSE: ${rmse_train:,.2f}")
print(f"  R² Score: {r2_train:.4f}")

print(f"\nOverfitting Check:")
if abs(r2_train - r2_test) < 0.1:
    print(f"  ✓ No significant overfitting. Training and test performance are similar.")
else:
    print(f"  ⚠ Possible overfitting. Training performance much better than test.")

# Step 7: Visualize Results
print("\n" + "="*70)
print("[STEP 7] Visualizing Results...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: All Data with Best-Fit Line
ax1 = axes[0, 0]
ax1.scatter(X, y, color='blue', alpha=0.6, s=100, label='Actual Data')

# Create a line for the best-fit line
X_line = np.array([[X.min()], [X.max()]])
y_line = model.predict(X_line)
ax1.plot(X_line, y_line, color='red', linewidth=3, label='Best-Fit Line')

ax1.set_xlabel('House Size (square feet)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
ax1.set_title('Linear Regression: House Price Prediction', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style='plain', axis='y')

# Plot 2: Actual vs Predicted (Test Set)
ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred, color='green', alpha=0.6, s=100)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Price ($)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Price ($)', fontsize=11, fontweight='bold')
ax2.set_title('Actual vs Predicted Prices (Test Set)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.ticklabel_format(style='plain', axis='both')

# Plot 3: Residuals (Errors)
ax3 = axes[1, 0]
residuals = y_test - y_test_pred
ax3.scatter(y_test_pred, residuals, color='purple', alpha=0.6, s=100)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Price ($)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
ax3.set_title('Residual Plot (Prediction Errors)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.ticklabel_format(style='plain', axis='both')

# Plot 4: Distribution of Residuals
ax4 = axes[1, 1]
ax4.hist(residuals, bins=8, color='orange', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('linear_regression_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'linear_regression_analysis.png'")
plt.show()

# Step 8: Summary and Insights
print("\n" + "="*70)
print("[SUMMARY] Key Findings")
print("="*70)

print(f"""
📊 EXPERIMENT SUMMARY:

1. MODEL EQUATION:
   Price = {model.coef_[0]:.2f} × Square Feet + {model.intercept_:,.2f}

2. KEY INSIGHTS:
   • Every 100 sq ft increase → ${model.coef_[0]*100:,.0f} price increase
   • Base price (intercept) → ${model.intercept_:,.0f}
   • Model R² Score → {r2_test:.4f} ({r2_test*100:.1f}% variance explained)

3. PREDICTION ERROR:
   • Average Error (RMSE) → ${rmse_test:,.0f}
   • Average Absolute Error → ${mae_test:,.0f}

4. MODEL QUALITY:
   • Training R²: {r2_train:.4f}
   • Testing R²: {r2_test:.4f}
   • Difference: {abs(r2_train - r2_test):.4f} (lower is better)

5. RECOMMENDATIONS:
""")

if r2_test > 0.8:
    print("   ✓ Model is very good! Ready for predictions.")
elif r2_test > 0.6:
    print("   ✓ Model is decent. Consider adding more features.")
else:
    print("   ⚠ Model could be improved. Try:")
    print("     - Adding more relevant features")
    print("     - Checking for data quality issues")
    print("     - Using polynomial regression")

print(f"""
6. NEXT STEPS:
   • Try polynomial regression for non-linear relationships
   • Add more features (age, location, etc.)
   • Test on new unseen data
   • Deploy model for real predictions

✅ Experiment 1 Complete! You now understand Linear Regression.
""")

print("="*70)
print("Ready for Experiment 2: Logistic Regression? 🚀")
print("="*70)
