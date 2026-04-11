"""
Coding Assessment
Topic: Unsupervised Learning, Clustering, and K-means
Python: 3.10+
Libraries: numpy>=1.24, scikit-learn>=1.3, matplotlib>=3.7
Dataset: Synthetic 2D datasets from sklearn.make_blobs and sklearn.make_moons
Student Name: _______________
Student ID:   _______________

Note:
- This assessment uses train/validation/test splits for a more careful workflow,
  even though clustering is often explored on the full dataset in small examples.
- Any implementation that matches the required behavior and validation checks
  should receive full credit, even if its internal code structure differs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# CONFIG
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
ARTIFACT_DIR = Path("artifacts")
PRIMARY_PLOT_NAME = "primary_clusters.png"
STRESS_PLOT_NAME = "stress_clusters.png"
AUDIT_PLOT_NAME = "audit_clusters.png"

ARTIFACT_DIR.mkdir(exist_ok=True)
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------
def prepare_dataset_splits() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate datasets, split them, and scale using train-only statistics."""
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

    Xp_train, Xp_temp, yp_train, yp_temp = train_test_split(
        X_primary_raw, y_primary_reference, test_size=0.30, random_state=RANDOM_STATE
    )
    Xp_val, Xp_test, yp_val, yp_test = train_test_split(
        Xp_temp, yp_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    Xs_train, Xs_temp, ys_train, ys_temp = train_test_split(
        X_stress_raw, y_stress_reference, test_size=0.30, random_state=RANDOM_STATE
    )
    Xs_val, Xs_test, ys_val, ys_test = train_test_split(
        Xs_temp, ys_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    scaler_primary = StandardScaler()
    scaler_stress = StandardScaler()

    primary = {
        "X_train": scaler_primary.fit_transform(Xp_train),
        "X_val": scaler_primary.transform(Xp_val),
        "X_test": scaler_primary.transform(Xp_test),
        "y_train_reference": yp_train,
        "y_val_reference": yp_val,
        "y_test_reference": yp_test,
    }
    stress = {
        "X_train": scaler_stress.fit_transform(Xs_train),
        "X_val": scaler_stress.transform(Xs_val),
        "X_test": scaler_stress.transform(Xs_test),
        "y_train_reference": ys_train,
        "y_val_reference": ys_val,
        "y_test_reference": ys_test,
    }
    return primary, stress


PRIMARY_DATA, STRESS_DATA = prepare_dataset_splits()


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
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=20)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=140)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / filename)
    plt.close()


def compute_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    diffs = X - centroids[labels]
    return float(np.sum(diffs * diffs))


def print_metric_block(title: str, metrics: dict[str, object]) -> None:
    print(f"\n[{title}]")
    for key, value in metrics.items():
        print(f"- {key}: {value}")


def summarize_cluster_balance(labels: np.ndarray) -> np.ndarray:
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
    # TODO [T1]: NumPy k-means from scratch
    # Bloom's Level: Apply
    # Difficulty: Easy / Medium  |  Expected lines: ~30
    # Description: Implement k-means using NumPy only. Alternate between point assignment
    # and centroid update until convergence or the maximum iteration limit is reached.
    # This directly tests the core algorithm from the lecture.
    # Hints: 1. Use squared Euclidean distance for assignment. 2. Broadcasting is a clean way
    # to build a distance matrix of shape (n_samples, k).
    # Expected behavior: Return integer labels of shape (n_samples,) and centroids of shape
    # (k, n_features). Labels should stay in the range [0, k-1].

    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T1 not yet implemented")
    # >>> END YOUR CODE <<<


def todo_t2_evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> dict[str, float]:
    # TODO [T2]: Compute clustering metrics
    # Bloom's Level: Apply
    # Difficulty: Easy  |  Expected lines: ~10
    # Description: Evaluate a clustering result using two internal metrics taught in the
    # lecture: inertia and silhouette score. This checks whether the algorithm output can
    # be connected to compactness and separation.
    # Hints: 1. Lower inertia means tighter clusters. 2. Reuse compute_inertia() and
    # sklearn.metrics.silhouette_score().
    # Expected behavior: Return a dict with keys 'inertia' and 'silhouette'. Silhouette
    # should lie between SILHOUETTE_MIN and SILHOUETTE_MAX.

    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T2 not yet implemented")
    # >>> END YOUR CODE <<<


def todo_t3_select_k(
    X_train: np.ndarray,
    X_val: np.ndarray,
    candidate_ks: tuple[int, ...] = K_PRIMARY_CANDIDATES,
) -> dict[str, object]:
    # TODO [T3]: Select K with validation data
    # Bloom's Level: Apply
    # Difficulty: Medium  |  Expected lines: ~20
    # Description: Train one clustering for each candidate K on the training split and use
    # the validation split to select the best K. This keeps model selection separate from
    # final checking and mirrors a more careful practical workflow.
    # Hints: 1. Fit centroids on X_train, then assign X_val points to the nearest centroid.
    # 2. Choose the K with the best validation silhouette score.
    # Expected behavior: Return a dict with keys 'best_k' and 'val_scores'. The selected K
    # must come from candidate_ks and val_scores length must match len(candidate_ks).

    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T3 not yet implemented")
    # >>> END YOUR CODE <<<


def todo_t4_compare_on_stress_dataset(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, float]:
    # TODO [T4]: Compare methods on a failure case
    # Bloom's Level: Analyze
    # Difficulty: Medium  |  Expected lines: ~20
    # Description: Compare k-means against an alternative clustering method on the moons
    # dataset, where centroid-based partitions are a poor fit. This tests the lecture claim
    # that k-means struggles on non-spherical clusters.
    # Hints: 1. Use K_STRESS for both methods. 2. Compare test-set silhouette scores after
    # assigning labels on X_test.
    # Expected behavior: Return a dict with 'kmeans_silhouette' and
    # 'agglomerative_silhouette'. The result should show a meaningful method difference.

    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T4 not yet implemented")
    # >>> END YOUR CODE <<<


def todo_t5_initialization_sensitivity(
    X: np.ndarray,
    k: int,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, object]:
    # TODO [T5]: Diagnose initialization sensitivity
    # Bloom's Level: Analyze
    # Difficulty: Hard  |  Expected lines: ~20
    # Description: Run k-means several times with different seeds and summarize how much the
    # inertia changes. This checks whether you can diagnose local-optimum sensitivity instead
    # of treating one run as definitive.
    # Hints: 1. One inertia value should be stored per seed. 2. Report min, max, and spread.
    # Expected behavior: Return a dict containing 'inertias', 'min_inertia', 'max_inertia',
    # and 'spread'. Length of inertias must equal len(seeds).

    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T5 not yet implemented")
    # >>> END YOUR CODE <<<


def todo_t6_cluster_audit_report(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    group_indicator: np.ndarray,
) -> dict[str, object]:
    # TODO [T6]: Design a clustering audit report
    # Bloom's Level: Create
    # Difficulty: Hard  |  Expected lines: ~25
    # Description: Build a compact audit report that combines clustering quality, cluster
    # balance, and group skew. Use group_indicator as a mock sensitive attribute and report
    # whether any cluster is dominated by one group.
    # Hints: 1. This is an audit summary, not a decision rule. 2. Build a cluster-by-group
    # table and compute the largest within-cluster group proportion.
    # Expected behavior: Return a dict with 'silhouette', 'cluster_sizes', 'group_table', and
    # 'max_group_proportion'. The last value must lie in [0.0, 1.0].

    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T6 not yet implemented")
    # >>> END YOUR CODE <<<


def todo_t7_improvement_suggestions(
    stress_metrics: dict[str, float],
    stability_summary: dict[str, object],
    audit_report: dict[str, object],
) -> dict[str, str]:
    # TODO [T7]: Propose pipeline improvements
    # Bloom's Level: Create
    # Difficulty: Medium  |  Expected lines: ~15
    # Description: Propose two or three concrete improvements to the clustering workflow using
    # evidence from earlier outputs. This adds a small design component beyond raw metric
    # reporting.
    # Hints: 1. Tie each suggestion to a specific issue such as non-spherical clusters,
    # initialization sensitivity, or group skew. 2. Return short, readable recommendation text.
    # Expected behavior: Return a dict with keys such as 'method_change', 'stability_change',
    # and 'audit_change'. Values should be short strings.

    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T7 not yet implemented")
    # >>> END YOUR CODE <<<


# ---------------------------------------------------------------------
# VALIDATION HARNESS
# ---------------------------------------------------------------------
def assign_to_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(int)


def validate_submission() -> None:
    checks: list[tuple[str, bool, str]] = []

    X_train = PRIMARY_DATA["X_train"]
    X_val = PRIMARY_DATA["X_val"]
    X_test = PRIMARY_DATA["X_test"]
    Xs_train = STRESS_DATA["X_train"]
    Xs_test = STRESS_DATA["X_test"]
    group_indicator = (X_test[:, 0] > np.median(X_test[:, 0])).astype(int)

    try:
        labels_1, centroids_1 = todo_t1_kmeans_numpy(X_train, K_PRIMARY_TRUE)
        labels_1_repeat, centroids_1_repeat = todo_t1_kmeans_numpy(X_train, K_PRIMARY_TRUE)
        ok = (
            isinstance(labels_1, np.ndarray)
            and isinstance(centroids_1, np.ndarray)
            and labels_1.shape == (X_train.shape[0],)
            and centroids_1.shape == (K_PRIMARY_TRUE, N_FEATURES)
            and np.array_equal(labels_1, labels_1_repeat)
            and np.allclose(centroids_1, centroids_1_repeat)
        )
        checks.append(("T1", ok, "shape and reproducibility"))
    except Exception as exc:
        checks.append(("T1", False, f"exception: {exc}"))
        centroids_1 = np.zeros((K_PRIMARY_TRUE, N_FEATURES))

    try:
        test_labels = assign_to_centroids(X_test, centroids_1)
        metrics_2 = todo_t2_evaluate_clustering(X_test, test_labels, centroids_1)
        ok = (
            isinstance(metrics_2, dict)
            and "inertia" in metrics_2
            and "silhouette" in metrics_2
            and metrics_2["inertia"] >= 0.0
            and SILHOUETTE_MIN <= metrics_2["silhouette"] <= SILHOUETTE_MAX
        )
        checks.append(("T2", ok, "metric keys and ranges"))
    except Exception as exc:
        checks.append(("T2", False, f"exception: {exc}"))

    try:
        result_3 = todo_t3_select_k(X_train, X_val)
        ok = (
            isinstance(result_3, dict)
            and "best_k" in result_3
            and "val_scores" in result_3
            and result_3["best_k"] in K_PRIMARY_CANDIDATES
            and len(result_3["val_scores"]) == len(K_PRIMARY_CANDIDATES)
        )
        checks.append(("T3", ok, "validation-based K selection"))
    except Exception as exc:
        checks.append(("T3", False, f"exception: {exc}"))

    try:
        result_4 = todo_t4_compare_on_stress_dataset(Xs_train, Xs_test)
        ok = (
            isinstance(result_4, dict)
            and "kmeans_silhouette" in result_4
            and "agglomerative_silhouette" in result_4
        )
        checks.append(("T4", ok, "stress comparison metrics"))
    except Exception as exc:
        checks.append(("T4", False, f"exception: {exc}"))

    try:
        result_5 = todo_t5_initialization_sensitivity(X_train, K_PRIMARY_TRUE)
        ok = (
            isinstance(result_5, dict)
            and len(result_5["inertias"]) == 5
            and result_5["spread"] >= 0.0
        )
        checks.append(("T5", ok, "initialization spread summary"))
    except Exception as exc:
        checks.append(("T5", False, f"exception: {exc}"))
        result_5 = {"spread": 0.0}

    try:
        test_labels = assign_to_centroids(X_test, centroids_1)
        result_6 = todo_t6_cluster_audit_report(X_test, test_labels, centroids_1, group_indicator)
        ok = (
            isinstance(result_6, dict)
            and "silhouette" in result_6
            and "cluster_sizes" in result_6
            and "group_table" in result_6
            and 0.0 <= result_6["max_group_proportion"] <= 1.0
        )
        checks.append(("T6", ok, "audit report structure"))
    except Exception as exc:
        checks.append(("T6", False, f"exception: {exc}"))
        result_6 = {"max_group_proportion": 0.0}

    try:
        result_7 = todo_t7_improvement_suggestions(
            {"kmeans_silhouette": 0.2, "agglomerative_silhouette": 0.4},
            result_5,
            result_6,
        )
        ok = isinstance(result_7, dict) and len(result_7) >= 2 and all(isinstance(v, str) for v in result_7.values())
        checks.append(("T7", ok, "improvement suggestions text"))
    except Exception as exc:
        checks.append(("T7", False, f"exception: {exc}"))

    print("\nValidation Results")
    for todo_id, passed, message in checks:
        print(f"- {todo_id}: {'PASS' if passed else 'FAIL'} ({message})")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    X_train = PRIMARY_DATA["X_train"]
    X_val = PRIMARY_DATA["X_val"]
    X_test = PRIMARY_DATA["X_test"]
    Xs_train = STRESS_DATA["X_train"]
    Xs_test = STRESS_DATA["X_test"]

    print("Implement the TODOs, then run this file to see the reports and checks.")

    labels_train, centroids = todo_t1_kmeans_numpy(X_train, K_PRIMARY_TRUE)
    labels_test = assign_to_centroids(X_test, centroids)
    metrics = todo_t2_evaluate_clustering(X_test, labels_test, centroids)
    k_selection = todo_t3_select_k(X_train, X_val)
    stress_result = todo_t4_compare_on_stress_dataset(Xs_train, Xs_test)
    stability = todo_t5_initialization_sensitivity(X_train, K_PRIMARY_TRUE)
    group_indicator = (X_test[:, 0] > np.median(X_test[:, 0])).astype(int)
    audit = todo_t6_cluster_audit_report(X_test, labels_test, centroids, group_indicator)
    suggestions = todo_t7_improvement_suggestions(stress_result, stability, audit)

    print_metric_block("T2 Metrics", metrics)
    print_metric_block("T3 Best K", k_selection)
    print_metric_block("T4 Stress Comparison", stress_result)
    print_metric_block("T5 Stability", stability)
    print_metric_block("T6 Audit", audit)
    print_metric_block("T7 Suggestions", suggestions)

    plot_clusters(X_test, labels_test, centroids, "Primary dataset clustering", PRIMARY_PLOT_NAME)
    validate_submission()


if __name__ == "__main__":
    main()
