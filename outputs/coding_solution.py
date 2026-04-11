"""
Coding Assessment
Topic: Unsupervised Learning, Clustering, and K-means
Python: 3.10+
Libraries: numpy>=1.24, scikit-learn>=1.3, matplotlib>=3.7
Dataset: Synthetic 2D datasets from sklearn.make_blobs and sklearn.make_moons

Benchmarks:
- Expected runtime: under 10 seconds on a typical 2-core CPU with 8 GB RAM.
- Expected silhouette range on the blob test split: roughly 0.55 to 0.90.
- Expected validation choice for K on the blob dataset: usually 3.
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


def assign_to_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(int)


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
    # We initialize k centroids by sampling data points without replacement.
    # Then we alternate between assignment and centroid update until the total
    # centroid movement is smaller than the convergence tolerance.
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    init_idx = rng.choice(n_samples, size=k, replace=False)
    centroids = X[init_idx].copy()

    for _ in range(max_iter):
        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = []
        for cluster_id in range(k):
            cluster_points = X[labels == cluster_id]
            if cluster_points.shape[0] == 0:
                new_centroids.append(centroids[cluster_id])
            else:
                new_centroids.append(np.mean(cluster_points, axis=0))
        new_centroids = np.vstack(new_centroids)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    final_labels = assign_to_centroids(X, centroids)
    return final_labels, centroids


def todo_t2_evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> dict[str, float]:
    # Inertia captures within-cluster compactness, while silhouette captures how
    # well separated the discovered clusters are from one another.
    inertia = compute_inertia(X, labels, centroids)
    silhouette = float(silhouette_score(X, labels))
    return {
        "inertia": round(inertia, REPORT_DECIMALS),
        "silhouette": round(silhouette, REPORT_DECIMALS),
    }


def todo_t3_select_k(
    X_train: np.ndarray,
    X_val: np.ndarray,
    candidate_ks: tuple[int, ...] = K_PRIMARY_CANDIDATES,
) -> dict[str, object]:
    # We fit one model per candidate K on the training split, then evaluate each
    # candidate on the validation split by assigning validation points to the learned centroids.
    val_scores: list[float] = []
    for k in candidate_ks:
        _, centroids = todo_t1_kmeans_numpy(X_train, k, seed=T1_INIT_SEED)
        val_labels = assign_to_centroids(X_val, centroids)
        val_score = float(silhouette_score(X_val, val_labels))
        val_scores.append(round(val_score, REPORT_DECIMALS))

    best_index = int(np.argmax(val_scores))
    best_k = int(candidate_ks[best_index])
    return {"best_k": best_k, "val_scores": val_scores}


def todo_t4_compare_on_stress_dataset(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, float]:
    # K-means is trained on the moons training split and evaluated on the moons test split.
    # Agglomerative clustering is fitted directly on the test split here as a simple comparison
    # baseline for the known non-spherical failure case.
    _, kmeans_centroids = todo_t1_kmeans_numpy(X_train, K_STRESS, seed=T1_INIT_SEED)
    kmeans_test_labels = assign_to_centroids(X_test, kmeans_centroids)

    agg = AgglomerativeClustering(n_clusters=K_STRESS)
    agg_test_labels = agg.fit_predict(X_test)

    kmeans_silhouette = float(silhouette_score(X_test, kmeans_test_labels))
    agglomerative_silhouette = float(silhouette_score(X_test, agg_test_labels))

    plot_clusters(X_test, kmeans_test_labels, kmeans_centroids, "Stress dataset: k-means", STRESS_PLOT_NAME)

    return {
        "kmeans_silhouette": round(kmeans_silhouette, REPORT_DECIMALS),
        "agglomerative_silhouette": round(agglomerative_silhouette, REPORT_DECIMALS),
    }


def todo_t5_initialization_sensitivity(
    X: np.ndarray,
    k: int,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, object]:
    # Repeating k-means with different seeds shows how much the solution depends on
    # initialization. We summarize this through the range of inertia values.
    inertias: list[float] = []
    for seed in seeds:
        labels, centroids = todo_t1_kmeans_numpy(X, k, seed=seed)
        inertias.append(round(compute_inertia(X, labels, centroids), REPORT_DECIMALS))

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
    # This audit report combines one quality metric with simple interpretability and group-skew
    # summaries. The goal is to surface potential concerns, not to automate any decision.
    silhouette = float(silhouette_score(X, labels))
    cluster_sizes = summarize_cluster_balance(labels).tolist()

    n_clusters = centroids.shape[0]
    group_values = np.unique(group_indicator)
    group_table = np.zeros((n_clusters, len(group_values)), dtype=int)

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        for j, group_id in enumerate(group_values):
            group_table[cluster_id, j] = int(np.sum(cluster_mask & (group_indicator == group_id)))

    max_group_proportion = 0.0
    for cluster_id in range(n_clusters):
        cluster_total = int(np.sum(group_table[cluster_id]))
        if cluster_total > 0:
            cluster_dominance = float(np.max(group_table[cluster_id]) / cluster_total)
            max_group_proportion = max(max_group_proportion, cluster_dominance)

    plot_clusters(X, labels, centroids, "Audit view: primary clustering", AUDIT_PLOT_NAME)

    return {
        "silhouette": round(silhouette, REPORT_DECIMALS),
        "cluster_sizes": cluster_sizes,
        "group_table": group_table.tolist(),
        "max_group_proportion": round(max_group_proportion, REPORT_DECIMALS),
    }


def todo_t7_improvement_suggestions(
    stress_metrics: dict[str, float],
    stability_summary: dict[str, object],
    audit_report: dict[str, object],
) -> dict[str, str]:
    # These suggestions are grounded in earlier outputs. Each recommendation points to a
    # specific weakness such as non-spherical structure, initialization sensitivity, or skew.
    method_change = (
        "Consider a non-centroid method such as agglomerative clustering or DBSCAN when the stress-case comparison "
        "shows that k-means handles curved structure poorly."
    )

    if float(stability_summary.get("spread", 0.0)) > 5.0:
        stability_change = "Increase n_init or use stronger centroid initialization because the inertia spread suggests sensitivity to starting points."
    else:
        stability_change = "Initialization looks fairly stable, but using multiple restarts remains a good default practice."

    if float(audit_report.get("max_group_proportion", 0.0)) > 0.80:
        audit_change = "Review feature choices and monitor group skew carefully because one cluster appears strongly dominated by a single group."
    else:
        audit_change = "Keep the audit step in the workflow so cluster quality is not interpreted without checking possible group skew."

    return {
        "method_change": method_change,
        "stability_change": stability_change,
        "audit_change": audit_change,
    }


# ---------------------------------------------------------------------
# VALIDATION HARNESS
# ---------------------------------------------------------------------
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
            labels_1.shape == (X_train.shape[0],)
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
        ok = metrics_2["inertia"] >= 0.0 and SILHOUETTE_MIN <= metrics_2["silhouette"] <= SILHOUETTE_MAX
        checks.append(("T2", ok, "metric keys and ranges"))
    except Exception as exc:
        checks.append(("T2", False, f"exception: {exc}"))

    try:
        result_3 = todo_t3_select_k(X_train, X_val)
        ok = result_3["best_k"] in K_PRIMARY_CANDIDATES and len(result_3["val_scores"]) == len(K_PRIMARY_CANDIDATES)
        checks.append(("T3", ok, "validation-based K selection"))
    except Exception as exc:
        checks.append(("T3", False, f"exception: {exc}"))

    try:
        result_4 = todo_t4_compare_on_stress_dataset(Xs_train, Xs_test)
        ok = all(key in result_4 for key in ("kmeans_silhouette", "agglomerative_silhouette"))
        checks.append(("T4", ok, "stress comparison metrics"))
    except Exception as exc:
        checks.append(("T4", False, f"exception: {exc}"))

    try:
        result_5 = todo_t5_initialization_sensitivity(X_train, K_PRIMARY_TRUE)
        ok = len(result_5["inertias"]) == 5 and result_5["spread"] >= 0.0
        checks.append(("T5", ok, "initialization spread summary"))
    except Exception as exc:
        checks.append(("T5", False, f"exception: {exc}"))
        result_5 = {"spread": 0.0}

    try:
        test_labels = assign_to_centroids(X_test, centroids_1)
        result_6 = todo_t6_cluster_audit_report(X_test, test_labels, centroids_1, group_indicator)
        ok = 0.0 <= result_6["max_group_proportion"] <= 1.0
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
        ok = len(result_7) >= 2 and all(isinstance(v, str) for v in result_7.values())
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
