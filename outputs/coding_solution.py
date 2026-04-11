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



def print_metric_block(title: str, metrics: dict[str, object]) -> None:
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
# TODO SOLUTIONS
# ---------------------------------------------------------------------
def todo_t1_kmeans_numpy(
    X: np.ndarray,
    k: int,
    max_iter: int = MAX_ITER,
    tol: float = TOLERANCE,
    seed: int = T1_INIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    # WHAT: Implement NumPy k-means from scratch.
    # WHY: This directly matches the lecture note's primary algorithm and avoids sklearn for the core logic.
    # WHY: Broadcasting keeps the implementation compact and makes the assignment step explicit.
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    init_idx = rng.choice(n_samples, size=k, replace=False)
    centroids = X[init_idx].copy()  # shape: (k, n_features)

    for _ in range(max_iter):
        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # shape: (n_samples, k)
        labels = np.argmin(distances, axis=1)  # shape: (n_samples,)

        new_centroids = []
        for cluster_id in range(k):
            cluster_points = X[labels == cluster_id]
            if cluster_points.shape[0] == 0:
                new_centroids.append(centroids[cluster_id])
            else:
                new_centroids.append(np.mean(cluster_points, axis=0))
        new_centroids = np.vstack(new_centroids)  # shape: (k, n_features)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    final_distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    final_labels = np.argmin(final_distances, axis=1)
    return final_labels.astype(int), centroids



def todo_t2_evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> dict[str, float]:
    # WHAT: Compute inertia and silhouette score.
    # WHY: These are the two internal evaluation ideas emphasized in the lecture note.
    # WHY: Returning both makes students compare compactness and separation rather than relying on one number.
    inertia = compute_inertia(X, labels, centroids)
    sil = float(silhouette_score(X, labels))
    return {
        "inertia": round(inertia, REPORT_DECIMALS),
        "silhouette": round(sil, REPORT_DECIMALS),
    }



def todo_t3_select_k(
    X: np.ndarray,
    candidate_ks: tuple[int, ...] = K_PRIMARY_CANDIDATES,
) -> dict[str, object]:
    # WHAT: Search over candidate K values using silhouette score.
    # WHY: The lecture note recommends trying multiple K values instead of assuming one value is correct.
    # WHY: Silhouette is used because it is taught and is more informative than inertia alone for selection.
    scores: list[float] = []
    for k in candidate_ks:
        labels, centroids = todo_t1_kmeans_numpy(X, k, seed=T1_INIT_SEED)
        sil = float(silhouette_score(X, labels))
        scores.append(round(sil, REPORT_DECIMALS))
    best_index = int(np.argmax(scores))
    best_k = int(candidate_ks[best_index])
    return {
        "best_k": best_k,
        "scores": scores,
    }



def todo_t4_compare_on_stress_dataset(X_stress: np.ndarray) -> dict[str, float]:
    # WHAT: Compare k-means and agglomerative clustering on non-spherical moons.
    # WHY: The lecture note explicitly says k-means struggles on curved or non-spherical clusters.
    # WHY: Agglomerative clustering is used as the alternative because it was taught as a related clustering family.
    kmeans_labels, kmeans_centroids = todo_t1_kmeans_numpy(X_stress, K_STRESS, seed=T1_INIT_SEED)
    agg = AgglomerativeClustering(n_clusters=K_STRESS)
    agg_labels = agg.fit_predict(X_stress)

    kmeans_silhouette = float(silhouette_score(X_stress, kmeans_labels))
    agglomerative_silhouette = float(silhouette_score(X_stress, agg_labels))

    plot_clusters(X_stress, kmeans_labels, kmeans_centroids, "Stress dataset: k-means", STRESS_PLOT_NAME)

    return {
        "kmeans_silhouette": round(kmeans_silhouette, REPORT_DECIMALS),
        "agglomerative_silhouette": round(agglomerative_silhouette, REPORT_DECIMALS),
    }



def todo_t5_initialization_sensitivity(
    X: np.ndarray,
    k: int,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, object]:
    # WHAT: Measure how inertia changes across random initializations.
    # WHY: The lecture note warns that different seeds can converge to different local minima.
    # WHY: A spread summary is easier to interpret than showing raw labels from multiple runs.
    inertias: list[float] = []
    for seed in seeds:
        labels, centroids = todo_t1_kmeans_numpy(X, k, seed=seed)
        inertia = compute_inertia(X, labels, centroids)
        inertias.append(round(float(inertia), REPORT_DECIMALS))

    min_inertia = round(float(np.min(inertias)), REPORT_DECIMALS)
    max_inertia = round(float(np.max(inertias)), REPORT_DECIMALS)
    spread = round(float(max_inertia - min_inertia), REPORT_DECIMALS)
    return {
        "inertias": inertias,
        "min_inertia": min_inertia,
        "max_inertia": max_inertia,
        "spread": spread,
    }



def todo_t6_cluster_audit_report(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    group_indicator: np.ndarray,
) -> dict[str, object]:
    # WHAT: Build a compact clustering audit report with metric, balance, and group-skew summary.
    # WHY: This synthesizes performance, interpretability, and ethical awareness from the lecture note.
    # WHY: A simple group table is transparent and avoids making any automated high-stakes decision.
    silhouette = float(silhouette_score(X, labels))
    cluster_sizes = summarize_cluster_balance(labels).tolist()

    n_clusters = centroids.shape[0]
    group_values = np.unique(group_indicator)
    group_table = np.zeros((n_clusters, group_values.shape[0]), dtype=int)
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        for group_id in group_values:
            group_table[cluster_id, int(group_id)] = int(np.sum(cluster_mask & (group_indicator == group_id)))

    max_group_proportion = 0.0
    for cluster_id in range(n_clusters):
        cluster_total = np.sum(group_table[cluster_id])
        if cluster_total > 0:
            cluster_max = np.max(group_table[cluster_id]) / cluster_total
            max_group_proportion = max(max_group_proportion, float(cluster_max))

    plot_clusters(X, labels, centroids, "Audit view: primary clustering", AUDIT_PLOT_NAME)

    return {
        "silhouette": round(silhouette, REPORT_DECIMALS),
        "cluster_sizes": cluster_sizes,
        "group_table": group_table.tolist(),
        "max_group_proportion": round(max_group_proportion, REPORT_DECIMALS),
    }


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
    except Exception as exc:
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
    except Exception as exc:
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
    except Exception as exc:
        checks.append(("T3", False, f"exception: {exc}"))

    try:
        result_4 = todo_t4_compare_on_stress_dataset(X_stress)
        ok = (
            isinstance(result_4, dict)
            and "kmeans_silhouette" in result_4
            and "agglomerative_silhouette" in result_4
        )
        checks.append(("T4", ok, "stress comparison metrics exist"))
    except Exception as exc:
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
    except Exception as exc:
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
    except Exception as exc:
        checks.append(("T6", False, f"exception: {exc}"))

    print("\nValidation Results")
    for todo_id, passed, message in checks:
        status = "PASS" if passed else "FAIL"
        print(f"- {todo_id}: {status} ({message})")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    # ── BENCHMARKS ──────────────────────────────────────────
    # Expected runtime: under 10 seconds on a 2-core CPU, 8GB RAM
    # T1 expected: primary clustering returns labels shape (300,) and centroids shape (3, 2)
    # T2 expected: silhouette on DATASET_PRIMARY typically around 0.65-0.85 after scaling
    # T3 expected: best_k should usually recover 3 on DATASET_PRIMARY
    # T4 expected: agglomerative_silhouette should be higher than kmeans_silhouette on make_moons
    # T5 expected: spread should be small on the blob dataset but not necessarily zero
    # T6 expected: report shows silhouette, cluster sizes, 3x2 group table, and max_group_proportion in [0, 1]
    # ────────────────────────────────────────────────────────
    X_primary = PRIMARY_DATA["X_scaled"]
    X_stress = STRESS_DATA["X_scaled"]
    group_indicator = (X_primary[:, 0] > np.median(X_primary[:, 0])).astype(int)

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
