# CSE 329 Lecture Note: Unsupervised Learning, Clustering, and K-means

## Section A: Knowledge Component

### 1. Foundational Concepts

In supervised learning, we learn from examples that already have labels. In unsupervised learning, those labels are missing, so the goal changes: instead of predicting a known answer, we try to uncover useful structure in the data. Clustering is one of the most common ways to do this, and K-means is often the first clustering algorithm students learn because it is simple, fast, and geometrically intuitive.

#### 1.1 Data matrix and features
A dataset is usually represented as a matrix **X** with:
- **n rows** = observations or samples
- **p columns** = features or variables

In supervised learning, we have both **X** and labels **y**. In unsupervised learning, we only have **X**.

#### 1.2 Distance and similarity
Clustering methods depend heavily on the idea that “similar” points should be grouped together.

Common distance measure:

- **Euclidean distance** between two points `x` and `z`:

\[
\|x-z\|_2 = \sqrt{\sum_{j=1}^{p}(x_j-z_j)^2}
\]

A smaller distance means the points are more similar under this metric.

#### 1.3 Mean and centroid
The **mean** of a set of points is their average. In clustering, the mean of all points assigned to a cluster is called the **centroid**.

If a cluster contains points \(x^{(1)}, x^{(2)}, \dots, x^{(m)}\), then its centroid is:

\[
\mu = \frac{1}{m}\sum_{i=1}^{m}x^{(i)}
\]

#### 1.4 Feature scaling
If one feature is measured in very large units and another in very small units, the large-scale feature can dominate distance calculations.

Example:
- income: 20,000 to 200,000
- age: 18 to 60

Without scaling, income may dominate clustering.

#### 1.5 Exploratory data analysis mindset
Unsupervised learning is often exploratory. We are not “predicting the correct answer” from labels. Instead, we are searching for structure, patterns, subgroups, or compact representations in the data.

---

### 2. Core Idea of Unsupervised Learning, Clustering, and K-means

#### 2.1 What problem does unsupervised learning solve?
Unsupervised learning deals with data that has **features but no labels**. Its goal is to discover hidden structure in the dataset.

Typical goals include:
- grouping similar observations
- finding unusual observations
- compressing high-dimensional data
- discovering latent patterns

#### 2.2 What problem does clustering solve?
Clustering aims to divide data into groups such that:
- points **within the same cluster** are similar
- points **in different clusters** are dissimilar

This differs from classification:
- **classification** uses known labels
- **clustering** creates groups from the data itself

#### 2.3 Why is K-means needed?
K-means is one of the simplest and most widely used clustering algorithms because it provides:
- a clear objective
- an intuitive geometric interpretation
- fast performance on many tabular datasets
- an effective baseline for exploratory analysis

It is especially useful when the clusters are roughly compact and separated.

---

### 3. Mechanism / How K-means Works

#### 3.1 Intuition
Imagine that we want to divide students into study-behavior groups using features such as:
- hours studied per week
- assignment submission delay
- class attendance

K-means tries to place **K representative centers** in the dataset and assign each point to the nearest center. Then it updates the centers to better represent the assigned points. This repeats until the assignments stabilize.

#### 3.2 Objective function
K-means tries to minimize the **within-cluster sum of squares** (WCSS), also called inertia.

If there are K clusters \(C_1, C_2, \dots, C_K\) with centroids \(\mu_1, \mu_2, \dots, \mu_K\), then K-means minimizes:

\[
\sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
\]

This means:
- each point should be close to its cluster center
- each cluster should be internally compact

#### 3.3 Step-by-step algorithm

**Input:** dataset X, number of clusters K

**Step 1: Initialize centroids**
- Randomly choose K starting centroids, or
- Use a smarter initialization such as **k-means++**

**Step 2: Assignment step**
- For each data point, compute its distance to each centroid
- Assign the point to the nearest centroid

**Step 3: Update step**
- For each cluster, recompute the centroid as the mean of all points assigned to that cluster

**Step 4: Repeat**
- Repeat the assignment and update steps until:
  - assignments stop changing, or
  - centroids move very little, or
  - a maximum number of iterations is reached

#### 3.4 Small example
Suppose we have 2D points representing customers:
- x-axis: annual spending on electronics
- y-axis: annual spending on groceries

If K = 3, K-means may produce clusters such as:
- budget customers
- family shoppers
- premium high-spending customers

Each group is represented by a centroid, which summarizes the typical customer in that cluster.

#### 3.5 Why it converges
At each iteration:
- the assignment step does not increase the objective
- the update step does not increase the objective

So the objective keeps decreasing or stays the same. However, K-means usually converges to a **local optimum**, not necessarily the global optimum.

---

### 4. Key Components / Building Blocks

#### 4.1 Number of clusters (K)
K is a user-defined parameter. Choosing K incorrectly can lead to misleading clusters.

- Too small K: different groups get merged
- Too large K: one natural group gets split unnecessarily

#### 4.2 Distance metric
Standard K-means uses **Euclidean distance**. This choice strongly shapes the behavior of the algorithm.

Because of this, K-means works best when clusters are:
- roughly spherical
- similar in spread
- separable by distance to the mean

#### 4.3 Initialization
Different starting centroids can produce different final solutions.

Common choices:
- random initialization
- k-means++ initialization
- multiple restarts, then keep the best run

#### 4.4 Centroid
A centroid is the mean of the points in a cluster. It is not required to be one of the original data points.

#### 4.5 Inertia / WCSS
This measures how tightly grouped the points are around their centroids.

Lower inertia usually means more compact clusters, but it does **not** automatically mean a better or more meaningful clustering.

#### 4.6 Cluster assignment
K-means performs **hard clustering**:
- each point belongs to exactly one cluster
- no partial memberships are allowed

---

### 5. Variants / Comparisons

#### 5.1 K-means vs Hierarchical Clustering

**K-means**
- requires K in advance
- creates flat clusters
- fast for larger datasets
- sensitive to initialization

**Hierarchical clustering**
- does not require specifying all clusters upfront in the same way
- builds a tree of merges or splits
- useful when we want multi-level grouping
- can be more computationally expensive

Use hierarchical clustering when the nested structure of groups matters.

#### 5.2 K-means vs DBSCAN

**K-means**
- assumes compact clusters
- requires K
- struggles with irregular shapes and outliers

**DBSCAN**
- density-based
- does not require K directly
- can find arbitrarily shaped clusters
- can label outliers as noise

Use DBSCAN when clusters are irregular or when outlier handling is important.

#### 5.3 K-means vs Gaussian Mixture Models (GMM)

**K-means**
- hard assignment
- centroid-based
- simple and fast

**GMM**
- soft assignment using probabilities
- model-based clustering
- can model ellipsoidal clusters better than K-means

Use GMM when overlap between clusters is expected and soft membership is useful.

#### 5.4 Variants of K-means
- **K-means++**: better centroid initialization
- **Mini-batch K-means**: scales to large datasets using small random batches
- **Bisecting K-means**: recursively splits clusters

---

### 6. Strengths and Limitations

#### 6.1 Strengths
- Easy to understand and implement
- Computationally efficient
- Works well on many clean tabular datasets
- Produces interpretable centroids
- Good baseline for exploratory analysis

#### 6.2 Limitations
- Must choose K beforehand
- Sensitive to initialization
- Sensitive to feature scaling
- Sensitive to outliers
- Poor performance on non-spherical clusters
- Assumes Euclidean geometry is meaningful
- Can produce misleading clusters even when no real clusters exist

#### 6.3 When it works well
K-means tends to work well when:
- features are properly scaled
- clusters are compact and reasonably separated
- outliers are limited
- the dataset is numeric and moderate to large in size

#### 6.4 When it fails
K-means often fails when:
- clusters have curved or irregular shapes
- clusters have very different sizes or densities
- there are many irrelevant/noisy features
- categorical data is encoded poorly and treated as Euclidean
- data contains strong outliers

---

### 7. Real-world Applications

#### 7.1 Customer segmentation
Businesses cluster customers using purchase history, frequency, spending level, and browsing behavior to support marketing and personalization.

#### 7.2 Document clustering
Documents can be grouped by topic using numerical text representations such as TF-IDF vectors or embeddings.

#### 7.3 Medical subtype discovery
Patient records or gene-expression measurements can reveal previously unknown disease subgroups.

#### 7.4 Image compression and color quantization
K-means can cluster similar pixel colors and replace each pixel with its cluster centroid to reduce the number of colors.

#### 7.5 Network traffic analysis
Unsupervised grouping can help identify typical patterns of device behavior and flag unusual segments for deeper investigation.

#### 7.6 Student learning analytics
Students can be grouped based on engagement, performance trends, and submission behavior to support intervention planning.

---

## Section B: Skill Component

This section turns the theory into a practical workflow using tabular data.

### 1. Dataset Handling (Tabular Data) and Preparation

Suppose we have a customer dataset with numerical features such as:
- annual income
- spending score
- average basket value
- number of visits per month

Typical workflow:
1. Load the dataset
2. Inspect column types
3. Remove duplicate rows if necessary
4. Handle missing values
5. Separate numeric and categorical columns
6. Select features appropriate for clustering

Important note: in clustering, feature choice is especially critical because the algorithm has no label to guide which features matter.

---

### 2. Preprocessing Steps

#### 2.1 Handle missing values
Common strategies:
- remove rows with too many missing values
- impute numeric features with median or mean
- impute categorical features with mode

#### 2.2 Encode categorical variables carefully
Standard K-means is designed for numeric data. If categorical variables exist:
- use encoding only if it makes semantic sense
- be cautious: one-hot encoded categorical variables can distort Euclidean distance

#### 2.3 Scale numerical features
This is usually essential.

Common choices:
- StandardScaler
- MinMaxScaler

#### 2.4 Detect extreme outliers
Outliers can drag centroids away from dense regions. Use boxplots, z-scores, or robust rules before final clustering.

---

### 3. Train-Validation-Test Split

Clustering is unsupervised, so splitting is different from supervised learning.

A practical strategy:
- **Train set**: fit preprocessing and clustering model
- **Validation set**: compare different K values and configurations
- **Test set**: check stability and generalization of the chosen configuration

Why split at all?
- to avoid tuning decisions on the full dataset
- to check whether the cluster structure remains stable on unseen data

A common workflow is:
1. Split raw data into train/validation/test
2. Fit imputers/scalers only on the training set
3. Transform validation and test using the fitted preprocessing
4. Fit K-means on the training set
5. Evaluate using internal metrics and cluster stability

---

### 4. Training Process (Python)

Below is a complete, runnable example.

**Python version:** 3.10+

**Recommended packages:**
- numpy >= 1.24
- pandas >= 2.0
- scikit-learn >= 1.3
- matplotlib >= 3.7
- seaborn >= 0.12

```python
# Python 3.10+
# Reproducible end-to-end K-means workflow for tabular data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

SEED = 42
np.random.seed(SEED)

# --------------------------------------------------
# 1. Create a synthetic tabular dataset
# --------------------------------------------------
X, _ = make_blobs(
    n_samples=900,
    centers=4,
    n_features=4,
    cluster_std=1.2,
    random_state=SEED,
)

columns = ["income_index", "spending_score", "visit_freq", "basket_value"]
df = pd.DataFrame(X, columns=columns)

# Inject a few missing values for demonstration
missing_rows = np.random.choice(df.index, size=20, replace=False)
missing_cols = np.random.choice(df.columns, size=20, replace=True)
for r, c in zip(missing_rows, missing_cols):
    df.loc[r, c] = np.nan

print("Dataset shape:", df.shape)
print(df.head())

# --------------------------------------------------
# 2. Train-validation-test split
# --------------------------------------------------
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

# --------------------------------------------------
# 3. Preprocessing pipeline
# --------------------------------------------------
numeric_features = columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features)
])

# Fit only on training data
X_train = preprocessor.fit_transform(train_df)
X_val = preprocessor.transform(val_df)
X_test = preprocessor.transform(test_df)

# --------------------------------------------------
# 4. Hyperparameter tuning over K
# --------------------------------------------------
candidate_k = range(2, 9)
results = []
models = {}

for k in candidate_k:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=SEED,
    )
    train_labels = model.fit_predict(X_train)
    val_labels = model.predict(X_val)

    train_inertia = model.inertia_
    train_silhouette = silhouette_score(X_train, train_labels)
    val_silhouette = silhouette_score(X_val, val_labels)

    results.append({
        "k": k,
        "train_inertia": train_inertia,
        "train_silhouette": train_silhouette,
        "val_silhouette": val_silhouette,
    })
    models[k] = model

results_df = pd.DataFrame(results)
print("\nTuning results:")
print(results_df)

# Select best K by validation silhouette
best_k = results_df.loc[results_df["val_silhouette"].idxmax(), "k"]
best_model = models[best_k]

print(f"\nSelected K = {best_k}")

# --------------------------------------------------
# 5. Final evaluation on test set
# --------------------------------------------------
train_labels = best_model.predict(X_train)
val_labels = best_model.predict(X_val)
test_labels = best_model.predict(X_test)

test_silhouette = silhouette_score(X_test, test_labels)
print(f"Test silhouette score: {test_silhouette:.4f}")

# --------------------------------------------------
# 6. Cluster centers in original feature space
# --------------------------------------------------
centers_scaled = best_model.cluster_centers_
scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]

# Since only numeric features are used, inverse transform directly
centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_original, columns=numeric_features)
print("\nCluster centers (approx. original feature scale):")
print(centers_df)

# --------------------------------------------------
# 7. Visualize clusters with PCA projection
# --------------------------------------------------
pca = PCA(n_components=2, random_state=SEED)
X_train_2d = pca.fit_transform(X_train)
centers_2d = pca.transform(centers_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=train_labels, s=18, alpha=0.7)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker="X", s=220, c="red")
plt.title(f"K-means on Training Data (K={best_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Stability check with a second run
# --------------------------------------------------
second_model = KMeans(
    n_clusters=int(best_k),
    init="k-means++",
    n_init=20,
    max_iter=300,
    random_state=99,
)
second_labels = second_model.fit_predict(X_train)

stability_ari = adjusted_rand_score(train_labels, second_labels)
print(f"Stability check (Adjusted Rand Index between two runs): {stability_ari:.4f}")
```

---

### 5. Evaluation Metrics

Because clustering has no labels, evaluation is harder than classification.

#### 5.1 Inertia (WCSS)
Measures compactness within clusters.

- Lower is better for compactness
- But inertia always tends to decrease as K increases
- So inertia alone is not enough

#### 5.2 Elbow method
Plot inertia against K.
Look for a point where improvement slows sharply.

Use it as a heuristic, not as a guaranteed answer.

#### 5.3 Silhouette score
For each point, compare:
- how close it is to its own cluster
- how far it is from the nearest other cluster

Range is approximately from -1 to 1:
- close to 1: good separation
- near 0: overlapping clusters
- negative: possible wrong assignment

#### 5.4 Cluster stability
Run the algorithm multiple times with different seeds or subsamples.
If the clustering changes drastically, the discovered structure may be unreliable.

#### 5.5 External evaluation (when labels exist only for analysis)
Sometimes a dataset has labels, but we purposely ignore them during clustering. Then external metrics can be used only for post-hoc analysis:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

These should not replace internal evaluation in a true unsupervised setting.

---

### 6. Basic Hyperparameter Tuning Strategy

Key hyperparameters for K-means:
- `n_clusters`
- `init`
- `n_init`
- `max_iter`

A reasonable tuning workflow:
1. Scale the features
2. Try a range of K values, such as 2 to 10
3. Compare inertia and silhouette score
4. Run each configuration with multiple initializations
5. Visualize results if possible
6. Prefer a solution that is both interpretable and stable

Practical advice:
- Use `init='k-means++'`
- Use sufficiently large `n_init` to reduce bad local optima
- Do not select K only because it gives the lowest inertia

---

### 7. Interpretation of Results

Clustering is not useful unless the clusters can be interpreted carefully.

#### 7.1 Examine cluster centers
Cluster centers summarize the typical profile of each group.

Example:
- Cluster 0: low spending, infrequent visits
- Cluster 1: medium spending, loyal frequent visits
- Cluster 2: premium high-spending customers

#### 7.2 Compare feature distributions per cluster
Use boxplots, means, medians, or grouped summaries.

#### 7.3 Check cluster sizes
Very tiny clusters may indicate:
- outliers
- poor K choice
- unusual but important niche segments

#### 7.4 Validate against domain knowledge
A mathematically compact cluster is not automatically useful. Ask:
- does this grouping make sense in the domain?
- is it actionable?
- is it stable over time?

---

### 8. Common Pitfalls and Debugging Strategies

The prompt requires discussion of overfitting, data leakage, class imbalance, and wrong metric choice. Some of these are naturally supervised-learning terms, but they still have meaningful analogues in clustering.

#### 8.1 Overfitting in clustering
In clustering, overfitting often appears as choosing too many clusters or discovering patterns caused by noise.

**Symptoms**
- very small clusters
- unstable assignments across runs
- clusters that do not generalize to validation/test data

**Debugging**
- reduce K
- remove noisy features
- test stability across random seeds
- inspect whether clusters remain meaningful on new data

#### 8.2 Data leakage
Leakage occurs when information from validation/test data influences preprocessing or model selection.

**Examples**
- scaling the full dataset before splitting
- selecting features after inspecting all data
- choosing K based on the test set

**Debugging**
- split first
- fit imputers/scalers only on training data
- reserve the test set for final checking only

#### 8.3 Class imbalance analogue in clustering
There are no class labels, but there can still be **cluster size imbalance**.

**Problem**
K-means may split large dense groups and ignore small meaningful groups, especially when clusters have different sizes or densities.

**Debugging**
- inspect cluster size distribution
- try DBSCAN or GMM if imbalance is severe
- reduce noisy dimensions
- check whether a small cluster is actually valuable rather than “bad”

#### 8.4 Wrong metric choice
This is one of the biggest mistakes.

**Examples**
- using Euclidean K-means on highly categorical data
- interpreting low inertia as proof of useful segmentation
- using clustering for a business decision without checking stability

**Debugging**
- match the method to the data geometry
- use multiple metrics
- visualize where possible
- compare with alternative clustering algorithms

#### 8.5 Unscaled features
**Problem**: one feature dominates distance.

**Fix**: standardize or normalize numerical features.

#### 8.6 Outliers
**Problem**: centroids get dragged by a few extreme points.

**Fix**: detect outliers first or try a more robust method.

#### 8.7 Irrelevant features
**Problem**: noise features hide true structure.

**Fix**: perform feature selection, PCA, or domain-informed pruning.

---

## Section C: Ethical Implications & Values

Ethics in clustering must be discussed specifically, because clustering directly creates groups that people may later treat as if they were real, objective categories.

### 1. Risks Specific to Unsupervised Learning, Clustering, and K-means

#### 1.1 Hidden grouping can become hidden discrimination
If K-means is used on human-related data, clusters can act as proxies for sensitive attributes even when race, gender, religion, or income group are not explicitly included.

Example:
- ZIP code, school type, browsing history, or purchasing patterns may indirectly encode socioeconomic status or ethnicity.

#### 1.2 False sense of objectivity
Because clustering is mathematical, decision-makers may assume the produced groups are natural facts rather than modeling choices.

But the result depends on:
- chosen features
- scaling method
- number of clusters K
- initialization
- preprocessing decisions

#### 1.3 Harm from forced grouping
K-means always assigns every point to a cluster, even when some points do not fit any meaningful group.

This can be harmful in practice because ambiguous individuals are still forced into a segment.

#### 1.4 Marginalized or rare groups may be erased
If a minority group is small or has lower density, K-means may merge it into a larger group or split it in misleading ways.

#### 1.5 Downstream misuse
Clusters created for exploration may later be used for:
- denying services
- targeted manipulation
- surveillance prioritization
- unequal allocation of educational or medical resources

---

### 2. Example Scenario Where Misuse Can Cause Harm

#### Scenario: University student risk segmentation
A university clusters students using:
- login frequency to the LMS
- assignment timing
- campus attendance signals
- library usage

The administration names one cluster “high dropout risk” and starts using it to:
- deny access to advanced courses
- intensify monitoring
- flag students for disciplinary review

**Possible harm**
- students with unstable internet access may be mislabeled
- commuting students or working students may appear “low engagement” even when they are performing well
- students from disadvantaged backgrounds may be disproportionately grouped into negative clusters

This harm occurs even though the clustering algorithm had no explicit harmful intent.

---

### 3. Technical Reasons Why These Risks Arise

#### 3.1 K-means depends on feature representation
The algorithm does not understand social meaning. It only sees distances in feature space. If the features embed historical inequality, the clusters may reproduce that inequality.

#### 3.2 Euclidean distance may be a poor model of human similarity
Two students or patients can be close numerically but different in context. K-means treats geometric closeness as meaningful similarity even when the domain does not justify it.

#### 3.3 Mandatory hard assignment
Every point must belong to exactly one cluster. There is no built-in “uncertain,” “mixed,” or “none of the above” category.

#### 3.4 Sensitivity to K and initialization
Different reasonable settings can produce different groupings. This means clusters can be unstable, yet organizations may still treat them as fixed truths.

#### 3.5 Small groups get overshadowed
The centroid objective tends to favor compact averages, which can bury rare but important patterns.

---

### 4. Responsible Practices

#### 4.1 Bias mitigation
- Audit whether clusters correlate strongly with sensitive or protected groups
- Remove or rethink features that act as problematic proxies
- Compare outcomes across demographic groups where lawful and appropriate
- Avoid naming clusters in stigmatizing ways such as “bad customers” or “weak students”

#### 4.2 Validation strategies
- test stability across seeds, time periods, and subsamples
- compare K-means with other methods such as DBSCAN or GMM
- use domain-expert review before operational deployment
- check whether clusters are actionable and not just mathematically convenient

#### 4.3 Human oversight
- treat clusters as hypotheses, not facts
- never use clustering alone for high-stakes decisions
- require human review before any intervention affecting rights, opportunities, or access

#### 4.4 Documentation and transparency
Document:
- which features were used
- how preprocessing was done
- why K was chosen
- what the limitations are
- what the clusters should **not** be used for

#### 4.5 Ongoing monitoring
If clustering is used in a live system, monitor whether:
- cluster sizes drift over time
- groups become less stable
- downstream decisions create inequitable impacts

---

## Summary

- Unsupervised learning works with data that has features but no labels.
- Clustering aims to group similar observations together.
- K-means is a centroid-based, hard clustering algorithm.
- It minimizes within-cluster squared distance to centroids.
- The algorithm alternates between assignment and centroid update steps.
- K-means works best for compact, roughly spherical clusters.
- It is sensitive to feature scaling, initialization, outliers, and the choice of K.
- Common evaluation tools include inertia, elbow plots, silhouette score, and stability checks.
- In practice, preprocessing and interpretation matter as much as running the algorithm.
- Ethically, clustering can create harmful or misleading groupings when used carelessly in human-centered applications.

## Exam Tips

- Clearly explain the difference between supervised learning and unsupervised learning.
- Be able to distinguish clustering from classification.
- Memorize the main K-means objective: minimize within-cluster sum of squared distances.
- Understand each step of the K-means algorithm: initialize, assign, update, repeat.
- Know why K-means is sensitive to initialization and why multiple restarts help.
- Be able to explain why scaling is important before using Euclidean-distance-based clustering.
- Compare K-means with hierarchical clustering, DBSCAN, and GMM in terms of assumptions and use cases.
- Do not claim that low inertia alone proves a good clustering.
- When asked about limitations, mention outliers, non-spherical clusters, unequal densities, and the need to choose K.
- For ethics questions, discuss forced grouping, proxy bias, instability, and the need for human oversight.
