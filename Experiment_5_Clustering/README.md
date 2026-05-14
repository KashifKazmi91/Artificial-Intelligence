# Experiment 5: Clustering & Unsupervised Learning

## 🎯 Learning Objectives

After completing this experiment, you will understand:
- What unsupervised learning is and how it differs from supervised
- K-Means clustering algorithm and how it works
- Hierarchical clustering methods
- Evaluating clusters without labels
- Dimensionality reduction (PCA)
- Finding optimal number of clusters
- Key concepts: centroid, silhouette score, elbow method

## 📖 Concept Explanation

### What is Unsupervised Learning?

Unlike supervised learning (labeled data), unsupervised learning finds patterns in **unlabeled data**:

**Supervised**: "Here are emails and their labels (spam/not spam). Learn to classify."
**Unsupervised**: "Here are emails. Group similar ones together."

### What is Clustering?

Clustering groups similar data points together:

**Real-world Examples**:
- **Customer Segmentation**: Group customers by behavior for targeted marketing
- **Document Clustering**: Group similar documents/articles
- **Gene Sequencing**: Group similar DNA sequences
- **Image Segmentation**: Group similar pixels in images
- **Anomaly Detection**: Identify unusual clusters

### K-Means Clustering

**How it works**:
```
1. Pick K random points as initial centroids
2. Assign each point to nearest centroid
3. Move centroids to center of their assigned points
4. Repeat steps 2-3 until centroids stop moving
```

**Visual Example**:
```
Iteration 1:       Iteration 2:       Final:
  * C1               * C1               * C1
   x x               x x                x x
  * C2             * C2              * C2
   o o               o o                o o
  * C3             * C3              * C3
```

Where:
- `*` = centroids (cluster centers)
- `x` = data points in cluster 1
- `o` = data points in cluster 2

### Key Concepts

#### Centroid
The center point of a cluster (average of all points in cluster)

#### Distance Metric
How we measure "closeness" (usually Euclidean distance)
```
distance = √((x₁-x₂)² + (y₁-y₂)²)
```

#### Inertia
Sum of squared distances from points to their centroid
- Lower inertia = better clustering
- Always decreases with more clusters

#### Silhouette Score
Measures how well-separated clusters are (-1 to 1)
- 1 = well-clustered
- 0 = overlapping
- -1 = misclassified

#### Elbow Method
Finding optimal K:
1. Try K = 1, 2, 3, 4, 5...
2. Calculate inertia for each
3. Plot inertia vs K
4. "Elbow" point = optimal K

### Hierarchical Clustering

Builds a tree of clusters:
```
Top:           All points together
                      |
               ---‾‾‾ | ‾‾‾---
              |           |
         Cluster 1    Cluster 2
              |           |
          pt1 pt2     pt3 pt4 pt5

Bottom:    Individual points
```

**Two approaches**:
1. **Agglomerative** (bottom-up): Start with points, merge similar ones
2. **Divisive** (top-down): Start with all, split into smaller clusters

### Dimensionality Reduction: PCA

Reduce data from many dimensions to 2-3 for visualization:

**Why?**
- Can't visualize >3D data
- Removes noise
- Speeds up clustering

**How it works**:
- Finds principal components (directions of maximum variance)
- Projects data onto these components
- Keeps most important information

## 🔧 Step-by-Step Code Explanation

### Step 1: Import Libraries
```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
```

### Step 2: Load and Prepare Data
```python
# Normalize features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Step 3: Elbow Method to Find Optimal K
```python
inertias = []
silhouette_scores = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot and find elbow
plt.plot(range(2, 10), inertias)
plt.xlabel('K (Number of Clusters)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

### Step 4: Train K-Means with Optimal K
```python
optimal_k = 3  # From elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
```

### Step 5: Evaluate Clustering
```python
# Silhouette score
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {sil_score:.3f}")

# Per-sample silhouette scores
sil_samples = silhouette_samples(X_scaled, cluster_labels)
```

### Step 6: Visualize Clusters
```python
# Use PCA to reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           marker='X', s=300, c='red', edgecolor='black')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-Means Clustering')
plt.show()
```

### Step 7: Analyze Clusters
```python
for i in range(optimal_k):
    cluster_points = X[cluster_labels == i]
    print(f"Cluster {i}: {len(cluster_points)} points")
    print(f"  Mean: {cluster_points.mean(axis=0)}")
```

## 📊 What You Should See

### 1. Elbow Plot
- Inertia decreases as K increases
- Elbow point = optimal K (knee in curve)

### 2. Silhouette Score
- 0.5-1.0: Good clustering
- 0.3-0.5: Okay clustering
- <0.3: Weak clustering

### 3. Cluster Visualization
- Distinct clusters with clear separation
- Centroids marked with 'X'
- Colors represent different clusters

### 4. Cluster Sizes
- Balanced clusters (similar sizes) usually better
- Very small or very large clusters may indicate issues

## 🧠 Common Issues & Solutions

### Issue 1: K-Means Stuck in Local Minimum
**Problem**: Different runs give different results

**Solution**:
```python
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
# n_init: Run algorithm 10 times with different initializations
```

### Issue 2: Can't Identify Optimal K
**Problem**: No clear elbow in plot

**Solutions**:
- Use silhouette method instead
- Domain knowledge (how many groups should exist?)
- Try multiple methods (hierarchical, DBSCAN)

### Issue 3: Features Have Different Scales
**Problem**: Larger features dominate distance calculation

**Solution**:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 💡 Key Takeaways

✅ Clustering finds patterns in unlabeled data
✅ K-Means iteratively improves cluster assignments
✅ Elbow method helps find optimal number of clusters
✅ Silhouette score quantifies clustering quality
✅ PCA enables visualization of high-dimensional data
✅ Always scale features before clustering
✅ Hierarchical clustering provides tree of solutions
✅ No single "correct" clustering - depends on goal

## 🚀 Next Steps

1. **Experiment with different K values**:
   - Try K = 2, 3, 4, 5, 6
   - Observe how clusters change
   - Calculate silhouette scores for each

2. **Try different clustering methods**:
   - Hierarchical clustering
   - DBSCAN (density-based)
   - Gaussian Mixture Models

3. **Try different datasets**:
   - Customer data
   - Gene expression data
   - Image pixel data

4. **Advanced topics**:
   - DBSCAN (handles arbitrary cluster shapes)
   - Gaussian Mixture Models (probabilistic)
   - Spectral clustering
   - Self-Organizing Maps

## 📚 Resources

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [K-Means Visualization](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
- [PCA Explained](https://en.wikipedia.org/wiki/Principal_component_analysis)

## ✅ Checklist Before Moving to Capstone

- [ ] Understand difference between supervised and unsupervised
- [ ] Know how K-Means algorithm works
- [ ] Can interpret silhouette scores
- [ ] Can use elbow method to find optimal K
- [ ] Understand what PCA does
- [ ] Can scale features before clustering
- [ ] Can visualize clusters in 2D
- [ ] Completed practice questions

Once you've checked all boxes, you're ready for the **Capstone Project: Multimodal AI Chat Application**! 🎉
