"""
Coding Assessment
Topic: Unsupervised Learning, Clustering, and K-means
Python: 3.14+
Libraries: numpy==1.24+, scikit-learn==1.8+, matplotlib==3.10+
Dataset: Synthetic unlabelled 2D clustering datasets generated with sklearn.make_blobs (primary, easy case) and sklearn.make_moons (stress case exposing non-spherical cluster limitation); no manual download needed
Student Name: _______________
Student ID:   _______________
"""

# ## Bootstrapped Coding Context
# 1. DATASET_STRATEGY:
#    - The topic is unsupervised learning, so the code operates on unlabelled data for fitting.
#    - The lecture note focuses on numeric tabular feature vectors and Euclidean-distance-based k-means.
#    - DATASET_PRIMARY: sklearn.datasets.make_blobs, because it produces compact centroid-like clusters that match the strengths of k-means.
#    - DATASET_STRESS: sklearn.datasets.make_moons, because it produces non-spherical curved clusters and exposes a known limitation taught in the lecture note.
# 2. CONFIG_VARIABLES:
#    - NUMBER OF CLUSTERS: K_PRIMARY_CANDIDATES, K_STRESS, K_AUDIT
#    - INITIALIZATION / RANDOMNESS: RANDOM_STATE, N_INIT, T1_INIT_SEED
#    - ITERATION CONTROL: MAX_ITER, TOLERANCE
#    - DATA CONSTANTS: N_SAMPLES_PRIMARY, N_SAMPLES_STRESS, N_FEATURES, CLUSTER_STD_PRIMARY, MOON_NOISE
#    - VISUALIZATION / REPORTING: FIGURE_WIDTH, FIGURE_HEIGHT, FIGURE_DPI
# 3. UTILITY_FUNCTIONS:
#    - prepare_datasets() -> tuple[dict, dict]: generate, scale, and package primary and stress datasets.
#    - plot_clusters(X, labels, centroids, title, filename) -> None: visualize 2D clustering assignments and optional centroids.
#    - compute_inertia(X, labels, centroids) -> float: compute within-cluster sum of squared distances.
#    - print_metric_block(title, metrics) -> None: print a formatted metric report.
#    - summarize_cluster_balance(labels) -> np.ndarray: count samples per cluster.
# 4. TODO_PLAN:
#    - T1 Apply (Easy-Medium): implement NumPy k-means from scratch on the primary dataset.
#    - T2 Apply (Easy): compute inertia and silhouette score for a clustering.
#    - T3 Apply (Medium): choose K by scanning candidate values with silhouette score.
#    - T4 Analyze (Medium): compare k-means against an alternative on make_moons to expose non-spherical-cluster failure.
#    - T5 Analyze (Hard): diagnose sensitivity to initialization through repeated runs and metric spread.
#    - T6 Create (Hard): design a clustering audit report that combines performance, stability, and ethical-risk checks.

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Version checks
# ---------------------------------------------------------------------
NUMPY_VERSION_PREFIX = "1."
SKLEARN_VERSION_PREFIX = "1."
MATPLOTLIB_VERSION_PREFIX = "3."

if not np.__version__.startswith(NUMPY_VERSION_PREFIX):
    print(f"[WARN] Expected numpy version starting with {NUMPY_VERSION_PREFIX}, got {np.__version__}")
if not sklearn.__version__.startswith(SKLEARN_VERSION_PREFIX):
    print(f"[WARN] Expected scikit-learn version starting with {SKLEARN_VERSION_PREFIX}, got {sklearn.__version__}")
if not matplotlib.__version__.startswith(MATPLOTLIB_VERSION_PREFIX):
    print(f"[WARN] Expected matplotlib version starting with {MATPLOTLIB_VERSION_PREFIX}, got {matplotlib.__version__}")

# ---------------------------------------------------------------------
# CONFIG BLOCK
# ---------------------------------------------------------------------
RANDOM_STATE = 7
T1_INIT_SEED = 11
MAX_ITER = 200
TOLERANCE = 1e-4
N_INIT = 10
N_SAMPLES_PRIMARY = 300
N_SAMPLES_STRESS = 300
N_FEATURES = 2
CLUSTER_STD_PRIMARY = 1.10
MOON_NOISE = 0.08
K_PRIMARY_TRUE = 3
K_STRESS = 2
K_AUDIT = 3
K_PRIMARY_CANDIDATES = (2, 3, 4, 5, 6)
FIGURE_WIDTH = 6
FIGURE_HEIGHT = 5
FIGURE_DPI = 120
REPORT_DECIMALS = 4
SILHOUETTE_MIN = -1.0
SILHOUETTE_MAX = 1.0
MIN_CLUSTER_SIZE = 1
ARTIFACT_DIR = Path("artifacts")
PRIMARY_PLOT_NAME = "primary_clusters.png"
STRESS_PLOT_NAME = "stress_clusters.png"
AUDIT_PLOT_NAME = "audit_clusters.png"

ARTIFACT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------
def prepare_datasets() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate and scale the primary and stress datasets.

    Returns:
        A pair of dictionaries. Each dictionary contains:
        - 'X_raw': original feature matrix, shape (n_samples, n_features)
        - 'X_scaled': standardized feature matrix, shape (n_samples, n_features)
        - 'y_reference': hidden reference labels for analysis only
    """
    X_primary_raw, y_primary_reference = make_blobs(
        n_samples=N_SAMPLES_PRIMARY,
        centers=K_PRIMARY_TRUE,
        n_features=N_FEATURES,
        cluster_std=CLUSTER_STD_PRIMARY,
        random_state=RANDOM_STATE,
    )
    X_stress_raw, y_stress_reference = make_moons(
        n_samples=N_SAMPLES_STRESS,
        noise=MOON_NOISE,
        random_state=RANDOM_STATE,
    )

    scaler_primary = StandardScaler()
    scaler_stress = StandardScaler()

    X_primary_scaled = scaler_primary.fit_transform(X_primary_raw)
    X_stress_scaled = scaler_stress.fit_transform(X_stress_raw)

    primary = {
        "X_raw": X_primary_raw,
        "X_scaled": X_primary_scaled,
        "y_reference": y_primary_reference,
    }
    stress = {
        "X_raw": X_stress_raw,
        "X_scaled": X_stress_scaled,
        "y_reference": y_stress_reference,
    }
    return primary, stress


PRIMARY_DATA, STRESS_DATA = prepare_datasets()


# ---------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------
def plot_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray | None,
    title: str,
    filename: str,
) -> None:
    """Plot 2D clustered data and optional centroids.

    Args:
        X: Feature matrix of shape (n_samples, 2).
        labels: Cluster labels of shape (n_samples,).
        centroids: Optional centroid matrix of shape (k, 2).
        title: Plot title.
        filename: Output filename saved inside ARTIFACT_DIR.

    Returns:
        None. Saves a PNG figure.
    """
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=20)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=140)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / filename)
    plt.close()



def compute_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Compute k-means inertia.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        labels: Cluster assignments, shape (n_samples,).
        centroids: Cluster centroids, shape (k, n_features).

    Returns:
        Float inertia value equal to the sum of squared distances.
    """
    diffs = X - centroids[labels]
    return float(np.sum(diffs * diffs))



def print_metric_block(title: str, metrics: dict[str, float | int | str]) -> None:
    """Print a small formatted metric report.

    Args:
        title: Block heading.
        metrics: Dictionary of metric names to values.

    Returns:
        None. Prints to stdout.
    """
    print(f"\n[{title}]")
    for key, value in metrics.items():
        print(f"- {key}: {value}")



def summarize_cluster_balance(labels: np.ndarray) -> np.ndarray:
    """Count members in each discovered cluster.

    Args:
        labels: Integer labels of shape (n_samples,).

    Returns:
        Array of counts per cluster in sorted label order.
    """
    return np.bincount(labels)


# ---------------------------------------------------------------------
# TODO SECTIONS
# ---------------------------------------------------------------------
def todo_t1_kmeans_numpy(
    X: np.ndarray,
    k: int,
    max_iter: int = MAX_ITER,
    tol: float = TOLERANCE,
    seed: int = T1_INIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    # ============================================================
    # TODO [T1]: NumPy k-means from scratch
    # Bloom's Level: Apply
    # Difficulty: Easy / Medium | Expected lines: ~30
    # Description: Implement the k-means algorithm using NumPy only.
    #   Alternate between assignment and centroid update until convergence.
    #   This directly tests the lecture note's primary algorithm.
    # Hints:
    #   1. Conceptual: Assign each point to the nearest centroid under squared Euclidean distance.
    #   2. Implementation: Use broadcasting to build a distance matrix of shape (n_samples, k).
    # Expected behavior: returns labels with shape (N_SAMPLES_PRIMARY,) and centroids with shape (k, N_FEATURES);
    #   labels should be integers in [0, k-1] and use MAX_ITER and TOLERANCE.
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T1 not yet implemented")
    # >>> END YOUR CODE <<<



def todo_t2_evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> dict[str, float]:
    # ============================================================
    # TODO [T2]: Compute clustering metrics
    # Bloom's Level: Apply
    # Difficulty: Easy | Expected lines: ~10
    # Description: Evaluate a clustering using the lecture note's main metrics.
    #   Compute inertia and silhouette score for the provided labels and centroids.
    #   This tests whether the student can connect algorithm output to evaluation.
    # Hints:
    #   1. Conceptual: Lower inertia is tighter; higher silhouette is better separated.
    #   2. Implementation: Reuse compute_inertia() and sklearn.metrics.silhouette_score().
    # Expected behavior: returns a dict with keys 'inertia' and 'silhouette';
    #   silhouette must lie between SILHOUETTE_MIN and SILHOUETTE_MAX.
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T2 not yet implemented")
    # >>> END YOUR CODE <<<



def todo_t3_select_k(
    X: np.ndarray,
    candidate_ks: tuple[int, ...] = K_PRIMARY_CANDIDATES,
) -> dict[str, object]:
    # ============================================================
    # TODO [T3]: Hyperparameter search for K
    # Bloom's Level: Apply
    # Difficulty: Medium | Expected lines: ~20
    # Description: Search over candidate values of K and choose the best one
    #   using silhouette score. This matches the lecture note's advice to combine
    #   internal evaluation with hyperparameter selection.
    # Hints:
    #   1. Conceptual: Fit one model per K and keep the K with the highest silhouette score.
    #   2. Implementation: Call your T1 function for each candidate K.
    # Expected behavior: returns a dict with keys 'best_k' and 'scores';
    #   best_k must be one of K_PRIMARY_CANDIDATES and scores length must match len(candidate_ks).
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T3 not yet implemented")
    # >>> END YOUR CODE <<<



def todo_t4_compare_on_stress_dataset(X_stress: np.ndarray) -> dict[str, float]:
    # ============================================================
    # TODO [T4]: Compare methods on a failure case
    # Bloom's Level: Analyze
    # Difficulty: Medium | Expected lines: ~20
    # Description: Compare k-means against an alternative clustering method on
    #   DATASET_STRESS. The lecture note says k-means struggles on non-spherical
    #   clusters; this TODO tests that idea empirically.
    # Hints:
    #   1. Conceptual: Use make_moons because nearest-centroid partitions are a poor fit.
    #   2. Implementation: Fit your NumPy k-means and sklearn AgglomerativeClustering, then compare silhouette scores.
    # Expected behavior: returns a dict with 'kmeans_silhouette' and 'agglomerative_silhouette';
    #   uses K_STRESS and should reveal a noticeable method difference on DATASET_STRESS.
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T4 not yet implemented")
    # >>> END YOUR CODE <<<



def todo_t5_initialization_sensitivity(
    X: np.ndarray,
    k: int,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, object]:
    # ============================================================
    # TODO [T5]: Diagnose initialization sensitivity
    # Bloom's Level: Analyze
    # Difficulty: Hard | Expected lines: ~20
    # Description: Run k-means multiple times with different seeds and summarize
    #   how much the inertia varies. This connects to the lecture note's warning
    #   that different initializations can reach different local minima.
    # Hints:
    #   1. Conceptual: Stable behavior means low spread across runs; unstable behavior means larger spread.
    #   2. Implementation: Store one inertia value per seed and report min, max, and spread.
    # Expected behavior: returns a dict containing 'inertias', 'min_inertia', 'max_inertia', and 'spread';
    #   length of inertias must equal len(seeds).
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T5 not yet implemented")
    # >>> END YOUR CODE <<<



def todo_t6_cluster_audit_report(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    group_indicator: np.ndarray,
) -> dict[str, object]:
    # ============================================================
    # TODO [T6]: Design a clustering audit report
    # Bloom's Level: Create
    # Difficulty: Hard | Expected lines: ~25
    # Description: Create a compact audit report that combines three ideas from
    #   the lecture note: clustering quality, stability/interpretability, and ethical
    #   awareness about skew across groups. Use the provided group_indicator as a mock
    #   sensitive attribute and report how unevenly groups are distributed across clusters.
    # Hints:
    #   1. Conceptual: This is an audit, not a classifier; summarize risk rather than making a decision.
    #   2. Implementation: Build a cluster-by-group table and compute the largest within-cluster group proportion.
    # Expected behavior: returns a dict with 'silhouette', 'cluster_sizes', 'group_table', and 'max_group_proportion';
    #   max_group_proportion must be between 0.0 and 1.0 and use K_AUDIT-related clustering output.
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T6 not yet implemented")
    # >>> END YOUR CODE <<<


# ---------------------------------------------------------------------
# VALIDATION HARNESS
# ---------------------------------------------------------------------
def validate_submission() -> None:
    """Run lightweight checks for each TODO and print PASS/FAIL."""
    checks: list[tuple[str, bool, str]] = []

    X_primary = PRIMARY_DATA["X_scaled"]
    X_stress = STRESS_DATA["X_scaled"]
    group_indicator = (X_primary[:, 0] > np.median(X_primary[:, 0])).astype(int)

    try:
        labels_1, centroids_1 = todo_t1_kmeans_numpy(X_primary, K_PRIMARY_TRUE)
        ok = (
            isinstance(labels_1, np.ndarray)
            and isinstance(centroids_1, np.ndarray)
            and labels_1.shape == (N_SAMPLES_PRIMARY,)
            and centroids_1.shape == (K_PRIMARY_TRUE, N_FEATURES)
            and np.issubdtype(labels_1.dtype, np.integer)
            and np.min(labels_1) >= 0
            and np.max(labels_1) < K_PRIMARY_TRUE
        )
        checks.append(("T1", ok, "labels/centroids shape and range"))
    except Exception as exc:  # pragma: no cover
        checks.append(("T1", False, f"exception: {exc}"))
        labels_1 = np.zeros(N_SAMPLES_PRIMARY, dtype=int)
        centroids_1 = np.zeros((K_PRIMARY_TRUE, N_FEATURES))

    try:
        metrics_2 = todo_t2_evaluate_clustering(X_primary, labels_1, centroids_1)
        ok = (
            isinstance(metrics_2, dict)
            and "inertia" in metrics_2
            and "silhouette" in metrics_2
            and metrics_2["inertia"] >= 0.0
            and SILHOUETTE_MIN <= metrics_2["silhouette"] <= SILHOUETTE_MAX
        )
        checks.append(("T2", ok, "metric keys and valid ranges"))
    except Exception as exc:  # pragma: no cover
        checks.append(("T2", False, f"exception: {exc}"))

    try:
        result_3 = todo_t3_select_k(X_primary)
        ok = (
            isinstance(result_3, dict)
            and "best_k" in result_3
            and "scores" in result_3
            and result_3["best_k"] in K_PRIMARY_CANDIDATES
            and len(result_3["scores"]) == len(K_PRIMARY_CANDIDATES)
        )
        checks.append(("T3", ok, "K search output structure"))
    except Exception as exc:  # pragma: no cover
        checks.append(("T3", False, f"exception: {exc}"))

    try:
        result_4 = todo_t4_compare_on_stress_dataset(X_stress)
        ok = (
            isinstance(result_4, dict)
            and "kmeans_silhouette" in result_4
            and "agglomerative_silhouette" in result_4
        )
        checks.append(("T4", ok, "stress comparison metrics exist"))
    except Exception as exc:  # pragma: no cover
        checks.append(("T4", False, f"exception: {exc}"))

    try:
        result_5 = todo_t5_initialization_sensitivity(X_primary, K_PRIMARY_TRUE)
        ok = (
            isinstance(result_5, dict)
            and "inertias" in result_5
            and "spread" in result_5
            and len(result_5["inertias"]) == 5
        )
        checks.append(("T5", ok, "initialization spread summary"))
    except Exception as exc:  # pragma: no cover
        checks.append(("T5", False, f"exception: {exc}"))

    try:
        result_6 = todo_t6_cluster_audit_report(X_primary, labels_1, centroids_1, group_indicator)
        ok = (
            isinstance(result_6, dict)
            and "silhouette" in result_6
            and "cluster_sizes" in result_6
            and "group_table" in result_6
            and "max_group_proportion" in result_6
            and 0.0 <= result_6["max_group_proportion"] <= 1.0
        )
        checks.append(("T6", ok, "audit report structure"))
    except Exception as exc:  # pragma: no cover
        checks.append(("T6", False, f"exception: {exc}"))

    print("\nValidation Results")
    for todo_id, passed, message in checks:
        status = "PASS" if passed else "FAIL"
        print(f"- {todo_id}: {status} ({message})")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    X_primary = PRIMARY_DATA["X_scaled"]
    X_stress = STRESS_DATA["X_scaled"]
    group_indicator = (X_primary[:, 0] > np.median(X_primary[:, 0])).astype(int)

    print("Run the TODOs after implementing them.")
    labels_1, centroids_1 = todo_t1_kmeans_numpy(X_primary, K_PRIMARY_TRUE)
    metrics_2 = todo_t2_evaluate_clustering(X_primary, labels_1, centroids_1)
    result_3 = todo_t3_select_k(X_primary)
    result_4 = todo_t4_compare_on_stress_dataset(X_stress)
    result_5 = todo_t5_initialization_sensitivity(X_primary, K_PRIMARY_TRUE)
    result_6 = todo_t6_cluster_audit_report(X_primary, labels_1, centroids_1, group_indicator)

    print_metric_block("T2 Metrics", metrics_2)
    print_metric_block("T3 Best K", result_3)
    print_metric_block("T4 Stress Comparison", result_4)
    print_metric_block("T5 Stability", result_5)
    print_metric_block("T6 Audit", result_6)

    plot_clusters(X_primary, labels_1, centroids_1, "Primary dataset clustering", PRIMARY_PLOT_NAME)
    validate_submission()


if __name__ == "__main__":
    main()
