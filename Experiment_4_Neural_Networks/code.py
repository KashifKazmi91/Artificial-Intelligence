"""
Experiment 4: Neural Networks

This script demonstrates neural network training using TensorFlow/Keras.
We'll classify handwritten digits from the MNIST dataset.

Conceptual Flow:
1. Load and explore MNIST dataset
2. Preprocess and normalize data
3. Build neural network model
4. Compile with loss function and optimizer
5. Train the model
6. Evaluate on test set
7. Make predictions
8. Visualize training history
9. Analyze misclassified examples
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EXPERIMENT 4: NEURAL NETWORKS")
print("Handwritten Digit Classification with MNIST")
print("="*70)

# Step 1: Load Dataset
print("\n[STEP 1] Loading MNIST Dataset...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"\nDataset Information:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Testing samples: {X_test.shape[0]}")
print(f"  Image shape: {X_train.shape[1:]} (28x28 pixels)")
print(f"  Classes: {len(np.unique(y_train))} (digits 0-9)")
print(f"  Data type: {X_train.dtype}")
print(f"\nPixel value range: {X_train.min()} to {X_train.max()}")
print(f"\nClass Distribution (Training):")
unique, counts = np.unique(y_train, return_counts=True)
for digit, count in zip(unique, counts):
    print(f"  Digit {digit}: {count} samples")

# Step 2: Preprocess Data
print("\n" + "="*70)
print("[STEP 2] Preprocessing Data...")
print("="*70)

print(f"\nOriginal data shape: {X_train.shape}")
print(f"Original pixel range: {X_train.min()} - {X_train.max()}")

# Normalize pixel values to 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"\nNormalized pixel range: {X_train.min():.3f} - {X_train.max():.3f}")
print(f"After normalization, all pixels are between 0 and 1")
print(f"\nWhy normalize?")
print(f"  1. Helps neural networks learn faster")
print(f"  2. Prevents numerical instability")
print(f"  3. Makes training more stable")

# Flatten images (28x28 -> 784)
X_train_flat = X_train.reshape(-1, 784)
X_test_flat = X_test.reshape(-1, 784)

print(f"\nFlattened shape: {X_train_flat.shape}")
print(f"Each image is now a vector of 784 values")
print(f"First 10 pixel values of first image: {X_train_flat[0][:10]}")

# Step 3: Build Neural Network
print("\n" + "="*70)
print("[STEP 3] Building Neural Network Model...")
print("="*70)

model = models.Sequential([
    # Input layer + First hidden layer
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),  # Prevent overfitting
    
    # Second hidden layer
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    # Third hidden layer
    layers.Dense(32, activation='relu'),
    
    # Output layer
    layers.Dense(10, activation='softmax')  # 10 classes (0-9)
])

print(f"\nModel Architecture:")
model.summary()

print(f"\nArchitecture Explanation:")
print(f"  Input: 784 neurons (28×28 pixel image flattened)")
print(f"  Layer 1: 128 neurons with ReLU activation")
print(f"    - ReLU: Introduces non-linearity")
print(f"    - Dropout(0.2): Randomly deactivate 20% of neurons during training")
print(f"  Layer 2: 64 neurons with ReLU activation")
print(f"    - Reduces dimensions, learns more abstract features")
print(f"  Layer 3: 32 neurons with ReLU activation")
print(f"    - Further dimension reduction")
print(f"  Output: 10 neurons with Softmax")
print(f"    - Softmax: Converts to probabilities (sum to 1)")
print(f"    - 10 classes: digits 0-9")

print(f"\nTotal Parameters: {model.count_params():,}")
print(f"  Trainable: Weights that get updated during training")

# Step 4: Compile Model
print("\n" + "="*70)
print("[STEP 4] Compiling Model...")
print("="*70)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # For integer labels (0-9)
    metrics=['accuracy']
)

print(f"\nOptimizer: Adam")
print(f"  - Learning rate: 0.001")
print(f"  - Why Adam? Adapts learning rate per parameter")
print(f"  - More efficient than standard SGD")

print(f"\nLoss Function: Sparse Categorical Crossentropy")
print(f"  - For multi-class classification with integer labels")
print(f"  - Measures difference between predicted and actual probability")

print(f"\nMetrics: Accuracy")
print(f"  - Tracks percentage of correct predictions")

# Step 5: Train Model
print("\n" + "="*70)
print("[STEP 5] Training Model...")
print("="*70)
print(f"\nStarting training (this may take a few minutes)...\n")

history = model.fit(
    X_train_flat, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,  # Use 10% of training data for validation
    verbose=1
)

print(f"\nTraining Complete!")
print(f"\nTraining Details:")
print(f"  Total epochs: 20")
print(f"  Batch size: 32")
print(f"  Validation split: 10%")
print(f"  Training samples used: {int(60000 * 0.9):,}")
print(f"  Validation samples: {int(60000 * 0.1):,}")

# Step 6: Evaluate on Test Set
print("\n" + "="*70)
print("[STEP 6] Evaluating Model on Test Set...")
print("="*70)

test_loss, test_accuracy = model.evaluate(X_test_flat, y_test, verbose=0)

print(f"\nTest Set Performance:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if test_accuracy > 0.95:
    print(f"\n  ✓ Excellent! Model performs very well.")
elif test_accuracy > 0.90:
    print(f"\n  ✓ Good performance! Model is working well.")
else:
    print(f"\n  ⚠ Could be improved. Consider:")
    print(f"    - More epochs")
    print(f"    - Larger network")
    print(f"    - Different learning rate")

# Training vs Test Comparison
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
overfitting_gap = final_train_acc - test_accuracy

print(f"\nGeneralization Analysis:")
print(f"  Final Training Accuracy: {final_train_acc:.4f}")
print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Train-Test Gap: {overfitting_gap:.4f}")

if overfitting_gap < 0.05:
    print(f"  ✓ Good generalization! No significant overfitting.")
elif overfitting_gap < 0.10:
    print(f"  ⚠ Slight overfitting. Consider adding more dropout.")
else:
    print(f"  ✗ Significant overfitting. Model memorized training data.")

# Step 7: Make Predictions
print("\n" + "="*70)
print("[STEP 7] Making Predictions on Test Set...")
print("="*70)

# Get predictions
y_pred_proba = model.predict(X_test_flat, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print(f"\nSample Predictions (First 10 test images):")
print(f"\n{'Image':>6} {'Actual':>7} {'Predicted':>11} {'Confidence':>12} {'Correct':<10}")
print("-" * 50)

for i in range(10):
    actual = y_test[i]
    predicted = y_pred[i]
    confidence = y_pred_proba[i][predicted]
    correct = "✓" if actual == predicted else "✗"
    print(f"{i:>6} {actual:>7} {predicted:>11} {confidence:>12.1%} {correct:<10}")

# Find some misclassified examples
misclassified_idx = np.where(y_pred != y_test)[0]
print(f"\nMisclassified Examples:")
print(f"  Total misclassified: {len(misclassified_idx)} out of {len(y_test)} ({len(misclassified_idx)/len(y_test)*100:.2f}%)")

if len(misclassified_idx) > 0:
    print(f"\n  First 5 Misclassified Examples:")
    for i in range(min(5, len(misclassified_idx))):
        idx = misclassified_idx[i]
        actual = y_test[idx]
        predicted = y_pred[idx]
        confidence = y_pred_proba[idx][predicted]
        print(f"    Image {idx}: Actual={actual}, Predicted={predicted} (Confidence={confidence:.1%})")

# Step 8: Confusion Matrix and Classification Report
print("\n" + "="*70)
print("[STEP 8] Detailed Classification Analysis...")
print("="*70)

cm = confusion_matrix(y_test, y_pred)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Step 9: Visualizations
print("\n" + "="*70)
print("[STEP 9] Creating Visualizations...")
print("="*70)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Training History - Accuracy
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history.history['accuracy'], 'b-', linewidth=2, label='Training')
ax1.plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.8, 1.0])

# Plot 2: Training History - Loss
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history.history['loss'], 'b-', linewidth=2, label='Training')
ax2.plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax2.set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Sample Predictions
ax3 = plt.subplot(2, 3, 3)
for i in range(9):
    ax_sub = plt.subplot(3, 3, i + 1)
    digit_image = X_test[i].reshape(28, 28)
    predicted = y_pred[i]
    actual = y_test[i]
    color = 'green' if actual == predicted else 'red'
    
    ax_sub.imshow(digit_image, cmap='gray')
    ax_sub.set_title(f'P:{predicted} A:{actual}', color=color, fontweight='bold')
    ax_sub.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=12, fontweight='bold', y=0.98)

# Plot 4: Confusion Matrix
ax4 = plt.subplot(2, 3, 4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax4,
            xticklabels=range(10), yticklabels=range(10))
ax4.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

# Plot 5: Per-Class Accuracy
ax5 = plt.subplot(2, 3, 5)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
ax5.bar(range(10), per_class_acc, color='skyblue', edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Digit', fontsize=11, fontweight='bold')
ax5.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax5.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
ax5.set_ylim([0.85, 1.0])
ax5.grid(True, alpha=0.3, axis='y')

for i, acc in enumerate(per_class_acc):
    ax5.text(i, acc + 0.005, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 6: Performance Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
MODEL PERFORMANCE SUMMARY

Test Accuracy: {test_accuracy:.2%}
Test Loss: {test_loss:.4f}

Misclassified: {len(misclassified_idx)} ({len(misclassified_idx)/len(y_test)*100:.2f}%)
Correctly Classified: {len(y_test) - len(misclassified_idx)} ({(1 - len(misclassified_idx)/len(y_test))*100:.2f}%)

Train-Test Gap: {overfitting_gap:.4f}
Generalization: {'✓ Good' if overfitting_gap < 0.05 else '⚠ Okay' if overfitting_gap < 0.10 else '✗ Poor'}

Total Parameters: {model.count_params():,}
Training Epochs: 20
Batch Size: 32

Best Per-Class: {target_names[np.argmax(per_class_acc)] if 'target_names' in dir() else np.argmax(per_class_acc)} ({per_class_acc.max():.2%})
Worst Per-Class: {target_names[np.argmin(per_class_acc)] if 'target_names' in dir() else np.argmin(per_class_acc)} ({per_class_acc.min():.2%})
"""

ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('neural_network_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'neural_network_analysis.png'")
plt.show()

# Step 10: Visualize Sample Misclassifications
if len(misclassified_idx) > 0:
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    
    for i in range(min(12, len(misclassified_idx))):
        idx = misclassified_idx[i]
        digit_image = X_test[idx].reshape(28, 28)
        actual = y_test[idx]
        predicted = y_pred[idx]
        confidence = y_pred_proba[idx][predicted]
        
        axes[i].imshow(digit_image, cmap='gray')
        axes[i].set_title(f'Actual: {actual}\nPredicted: {predicted}\nConfidence: {confidence:.1%}',
                         color='red', fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('misclassified_examples.png', dpi=300, bbox_inches='tight')
    print("✓ Misclassified examples saved as 'misclassified_examples.png'")
    plt.show()

# Step 11: Summary
print("\n" + "="*70)
print("[SUMMARY] Key Findings")
print("="*70)

print(f"""
📊 EXPERIMENT SUMMARY:

1. NEURAL NETWORK ARCHITECTURE:
   Input Layer: 784 neurons (28×28 pixel images)
   Hidden Layer 1: 128 neurons (ReLU) + Dropout(0.2)
   Hidden Layer 2: 64 neurons (ReLU) + Dropout(0.2)
   Hidden Layer 3: 32 neurons (ReLU)
   Output Layer: 10 neurons (Softmax for 10 classes)
   Total Parameters: {model.count_params():,}

2. TRAINING RESULTS:
   Final Training Accuracy: {final_train_acc:.4f}
   Final Validation Accuracy: {final_val_acc:.4f}
   Test Accuracy: {test_accuracy:.4f}
   Test Loss: {test_loss:.4f}

3. GENERALIZATION:
   Train-Test Gap: {overfitting_gap:.4f}
   Status: {'✓ Excellent' if overfitting_gap < 0.02 else '✓ Good' if overfitting_gap < 0.05 else '⚠ Moderate' if overfitting_gap < 0.10 else '✗ Poor'}

4. CLASSIFICATION PERFORMANCE:
   Correctly Classified: {len(y_test) - len(misclassified_idx):,} / {len(y_test):,}
   Misclassified: {len(misclassified_idx):,} / {len(y_test):,}
   Error Rate: {len(misclassified_idx)/len(y_test)*100:.2f}%

5. PER-CLASS PERFORMANCE:
   Best Accuracy: {target_names[np.argmax(per_class_acc)] if 'target_names' in dir() else np.argmax(per_class_acc)} ({per_class_acc.max():.2%})
   Worst Accuracy: {target_names[np.argmin(per_class_acc)] if 'target_names' in dir() else np.argmin(per_class_acc)} ({per_class_acc.min():.2%})

6. KEY INSIGHTS:
   ✓ Neural networks successfully learned digit patterns
   ✓ Dropout helped prevent overfitting
   ✓ Softmax activation produced good probability estimates
   ✓ Adam optimizer converged efficiently

7. POTENTIAL IMPROVEMENTS:
   • Use Convolutional Neural Network (CNN) for images
   • Increase model capacity (more neurons/layers)
   • Data augmentation (rotate, shift, scale images)
   • Experiment with different learning rates
   • Use batch normalization
   • Implement learning rate scheduling

✅ Experiment 4 Complete! You now understand Neural Networks.
""")

print("="*70)
print("Ready for Experiment 5: Clustering? 🚀")
print("="*70)
