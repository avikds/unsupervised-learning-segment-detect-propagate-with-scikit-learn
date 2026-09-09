"""
Unsupervised Learning: Segment, Detect, Propagate with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - blobs_data
from sklearn.datasets import make_blobs

def blobs_data(random_state=42):
    centers = [[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]]
    
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state
    )
    
    return X, y

# Step 2 - fit_kmeans
from sklearn.cluster import KMeans

def fit_kmeans(X, k, random_state=42):
    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state
    )
    return km.fit(X)

# Step 3 - inertia_curve
def inertia_curve(X, ks):
    return {k: float(fit_kmeans(X, k).inertia_) for k in ks}

# Step 4 - silhouette_curve
from sklearn.metrics import silhouette_score

def silhouette_curve(X, ks):
    return {
        k: float(silhouette_score(X, fit_kmeans(X, k).labels_))
        for k in ks
        if k >= 2
    }

def best_k_by_silhouette(curve):
    return min(curve, key=lambda k: (-curve[k], k))

# Step 5 - fit_dbscan
from sklearn.cluster import DBSCAN

def fit_dbscan(X, eps=0.2, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X)

def dbscan_summary(dbscan):
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    n_core = len(dbscan.core_sample_indices_)

    return {
        "n_clusters": int(n_clusters),
        "n_noise": n_noise,
        "n_core": int(n_core)
    }

# Step 6 - dbscan_predict
from sklearn.neighbors import KNeighborsClassifier

def dbscan_predict(dbscan, X_new, n_neighbors=50):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]
    knn.fit(dbscan.components_, core_labels)

    return knn.predict(X_new).astype(int)

# Step 7 - fit_gmm
from sklearn.mixture import GaussianMixture

def fit_gmm(X, n_components, random_state=42):
    return GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state
    ).fit(X)

def bic_curve(X, ks):
    return {
        k: float(fit_gmm(X, k).bic(X))
        for k in ks
    }

# Step 8 - flag_anomalies
import numpy as np

def flag_anomalies(gmm, X, contamination=0.04):
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)
    return densities < threshold

# Step 9 - digits_data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

# Step 10 - baseline_50_random
from sklearn.linear_model import LogisticRegression

def baseline_50_random(X_train, y_train, X_test, y_test, n_labeled=50, random_state=42):
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train[:n_labeled], y_train[:n_labeled])
    
    return float(clf.score(X_test, y_test))

# Step 11 - representative_digits
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state)
    distances = kmeans.transform(X_train)
    rep_idx = np.array(
        [np.argmin(distances[:, j]) for j in range(k)],
        dtype=int
    )

    return kmeans, rep_idx

# Step 12 - train_on_representatives
def train_on_representatives(X_train, y_train, rep_idx, X_test, y_test):
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train[rep_idx], y_train[rep_idx])

    return float(clf.score(X_test, y_test))

# Step 13 - propagate_and_train
def propagate_and_train(X_train, y_train, kmeans, rep_idx, X_test, y_test, percentile=20):
    distances = kmeans.transform(X_train)
    cluster_labels = kmeans.predict(X_train)

    propagated_labels = np.empty(len(X_train), dtype=int)
    selected = np.zeros(len(X_train), dtype=bool)

    for j in range(kmeans.n_clusters):
        cluster_mask = cluster_labels == j
        cluster_indices = np.where(cluster_mask)[0]
        cluster_distances = distances[cluster_indices, j]

        threshold = np.percentile(cluster_distances, percentile)
        selected_cluster = cluster_distances <= threshold

        selected[cluster_indices[selected_cluster]] = True
        propagated_labels[cluster_indices] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_propagated = propagated_labels[selected]

    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_selected, y_propagated)

    label_accuracy = float(np.mean(y_propagated == y_train[selected]))
    test_accuracy = float(clf.score(X_test, y_test))

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy
    }

# Step 14 - synthetic_image
def synthetic_image(size=48):
    img = np.zeros((size, size, 3), dtype=np.float64)

    half = size // 2
    gradient = np.linspace(0, 1, half)

    # Left half: red -> yellow
    img[:, :half, 0] = 1.0
    img[:, :half, 1] = gradient

    # Right half: blue -> cyan
    img[:, half:, 1] = gradient[:size - half]
    img[:, half:, 2] = 1.0

    # White square in the top-left corner
    square = size // 4
    img[:square, :square] = 1.0

    return img

# Step 15 - segment_colors (not yet solved)
# TODO: implement

# Step 16 - save_and_reload_clusterer (not yet solved)
# TODO: implement

# Step 17 - predict_digit_labels (not yet solved)
# TODO: implement

