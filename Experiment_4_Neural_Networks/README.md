# Experiment 4: Neural Networks - Introduction to Deep Learning

## 🎯 Learning Objectives

After completing this experiment, you will understand:
- What artificial neurons are and how they work
- Neural network architecture (layers, neurons, weights)
- Forward propagation and backpropagation
- Activation functions and their purposes
- How neural networks learn through training
- Loss functions and optimization
- Key concepts: perceptron, layers, activation, backprop

## 📖 Concept Explanation

### What is a Neuron?

An artificial neuron mimics biological neurons:

```
Inputs → [Weights] → [Sum] → [Activation] → Output
  x1  →    w1   \               f(z)      →  ŷ
  x2  →    w2   →→ z = Σ(x*w)+b →        →
  x3  →    w3   /                        →
```

**Mathematical Formula**:
```
z = w1*x1 + w2*x2 + w3*x3 + b
output = f(z)  (where f is activation function)
```

### Neural Network Architecture

**Layers**:
- **Input Layer**: Takes raw data
- **Hidden Layers**: Process data, learn patterns
- **Output Layer**: Makes final predictions

**Example**:
```
Input Layer    Hidden Layer    Output Layer
   x1              h1              ŷ
    ●─────────────●
   x2    weights   h2    weights    ●
    ●─────────────●
   x3              h3
    ●─────────────●
```

### Activation Functions

Why do we need them? Without them, stacking layers wouldn't help (would still be linear).

**Common Activation Functions**:

1. **ReLU (Rectified Linear Unit)**
   - Formula: f(x) = max(0, x)
   - Most popular for hidden layers
   - Efficient and works well

2. **Sigmoid**
   - Formula: f(x) = 1 / (1 + e^-x)
   - Output between 0 and 1
   - Used in output layer for binary classification

3. **Tanh**
   - Formula: f(x) = (e^x - e^-x) / (e^x + e^-x)
   - Output between -1 and 1
   - Often better than sigmoid

4. **Softmax**
   - Converts outputs to probabilities (sum to 1)
   - Used in output layer for multi-class classification

### Forward Propagation

How predictions are made:
```
1. Input data enters
2. Multiplied by weights in hidden layer 1
3. Add bias, apply activation
4. Result goes to hidden layer 2
5. Repeat process
6. Output layer produces final prediction
```

### Backpropagation

How the network learns:
```
1. Make prediction (forward pass)
2. Calculate error (loss)
3. Calculate how much each weight contributed to error
4. Update weights to reduce error (gradient descent)
5. Repeat until error is minimal
```

### Key Concepts

**Loss Function**: Measures prediction error
- Mean Squared Error (MSE): For regression
- Cross-entropy: For classification

**Optimizer**: Updates weights
- SGD: Stochastic Gradient Descent
- Adam: Adaptive Moment Estimation (most popular)
- RMSprop: Root Mean Square Propagation

**Learning Rate**: How big the weight updates are
- Too small: Slow learning
- Too large: Overshoots and diverges
- Good range: 0.001 to 0.01

**Epochs**: How many times to go through all data
- More epochs = more learning (but risk of overfitting)
- Usually 10-100 epochs

**Batch Size**: How many samples to process before updating weights
- Smaller batches: More updates, noisier
- Larger batches: Fewer updates, more stable

## 🔧 Step-by-Step Code Explanation

### Step 1: Import Libraries
```python
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
```

### Step 2: Prepare Data
```python
# Normalize pixel values to 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten 28x28 images to 784 values
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
```

### Step 3: Build Neural Network
```python
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),                    # Prevent overfitting
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes
])
```

**Architecture Explanation**:
- Input: 784 (28×28 pixels)
- Hidden 1: 128 neurons with ReLU
- Dropout: Randomly deactivate 20% during training
- Hidden 2: 64 neurons with ReLU
- Dropout: 20% again
- Hidden 3: 32 neurons with ReLU
- Output: 10 neurons (digits 0-9) with Softmax

### Step 4: Compile Model
```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # For integer labels
    metrics=['accuracy']
)
```

### Step 5: Train Model
```python
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,  # Use 10% of training for validation
    verbose=1
)
```

### Step 6: Evaluate Model
```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### Step 7: Make Predictions
```python
predictions = model.predict(X_test[:10])
predicted_classes = np.argmax(predictions, axis=1)
```

### Step 8: Visualize Training
```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'])
plt.show()
```

## 📊 What You Should See

### 1. Training Progress
```
Epoch 1/20
1875/1875 [==============================] - 4s 2ms/step
loss: 0.4532 - accuracy: 0.8645 - val_loss: 0.1234 - val_accuracy: 0.9621

Epoch 2/20
loss: 0.1234 - accuracy: 0.9632 - val_loss: 0.0876 - val_accuracy: 0.9721
```

Accuracy should increase and loss should decrease each epoch.

### 2. Final Accuracy
- Good: > 95%
- Excellent: > 97%
- With good hyperparameters: > 98%

### 3. Loss Curves
- Training loss: Continuously decreases
- Validation loss: Decreases then plateaus
- If gap widens: Overfitting

## 🧠 Common Issues & Solutions

### Issue 1: Accuracy Not Improving
**Causes**:
- Learning rate too small
- Model too simple
- Poor data preprocessing

**Solutions**:
```python
# Increase learning rate
optimizer = Adam(learning_rate=0.01)  # was 0.001

# Add more layers
layers.Dense(256, activation='relu'),
layers.Dense(128, activation='relu'),

# Better preprocessing
X = (X - X.mean()) / X.std()
```

### Issue 2: Overfitting
**Signs**: Training accuracy high, validation accuracy low

**Solutions**:
```python
# Add dropout
layers.Dropout(0.5),  # Increase from 0.2

# Add regularization
layers.Dense(128, activation='relu', 
            kernel_regularizer=keras.regularizers.l2(0.01)),

# Use early stopping
from tensorflow.keras.callbacks import EarlyStopping
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]
model.fit(..., callbacks=callbacks)
```

### Issue 3: Training Too Slow
**Solutions**:
```python
# Larger batch size
batch_size=128  # was 32

# Fewer layers/neurons
layers.Dense(64, activation='relu'),  # was 128

# Use GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## 💡 Key Takeaways

✅ Neural networks learn patterns through layers of neurons
✅ Activation functions enable learning of non-linear relationships
✅ Forward pass makes predictions, backprop updates weights
✅ Loss functions measure error, optimizers minimize it
✅ More layers = more complex patterns (but risk of overfitting)
✅ Dropout prevents overfitting by randomly disabling neurons
✅ Learning rate controls size of weight updates
✅ Validation set helps detect overfitting

## 🚀 Next Steps

1. **Experiment with architecture**:
   - Try different layer sizes: 64, 128, 256, 512
   - Try different depths: 2, 3, 4, 5 hidden layers
   - Observe impact on accuracy and training time

2. **Try different datasets**:
   - [CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10) - Color images
   - [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) - Clothing images
   - [Iris/Wine](https://scikit-learn.org/stable/datasets/) - Tabular data

3. **Advanced topics**:
   - Convolutional Neural Networks (CNN)
   - Recurrent Neural Networks (RNN)
   - Transfer learning
   - Custom training loops

## 📚 Resources

- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide)
- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (Video series)
- [Stanford CS231n CNN Course](http://cs231n.github.io/)

## ✅ Checklist Before Moving to Experiment 5

- [ ] Understand what neurons do and how they work
- [ ] Know the difference between activation functions
- [ ] Understand forward propagation
- [ ] Understand backpropagation basics
- [ ] Know what dropout does and why it helps
- [ ] Can build and train a neural network
- [ ] Can interpret training curves
- [ ] Can detect and fix overfitting
- [ ] Completed practice questions

Once you've checked all boxes, you're ready for **Experiment 5: Clustering**! 🎉
