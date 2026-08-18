"""
Step 3 — PCA + K-Means Clustering
Mobile Churn Prediction (Maharashtra Cities)

Takes data/features_matrix.csv (Step 2 output — already imputed, scaled,
encoded) and:
  1. Runs PCA to see how much variance a handful of components capture,
     and to get a 2D view we can actually plot.
  2. Runs k-means to segment customers, choosing k via the elbow method
     + silhouette score rather than guessing.
  3. Profiles each cluster in plain, original-unit terms and against the
     actual churn rate, for the written summary.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

features = pd.read_csv("data/features_matrix.csv")
readable = pd.read_csv("data/features_readable.csv")

X = features.drop(columns=["churn"]).values
y = features["churn"].values
print(f"Feature matrix for clustering/PCA: {X.shape}")

# 1. PCA

pca_full = PCA(random_state=42).fit(X)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = int(np.argmax(cum_var >= 0.95) + 1)
print(f"Components needed for 95% variance: {n_components_95} "
      f"(out of {X.shape[1]} original dimensions)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o", ms=3)
ax.axhline(0.95, color="red", linestyle="--", label="95% variance")
ax.axvline(n_components_95, color="gray", linestyle=":")
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("PCA — explained variance vs. number of components")
ax.legend()
fig.tight_layout()
fig.savefig("plots/pca_explained_variance.png", dpi=110)
plt.close(fig)

# 2D PCA purely for visualization
pca_2d = PCA(n_components=2, random_state=42)
X_pca2 = pca_2d.fit_transform(X)
print(f"2D PCA explains {pca_2d.explained_variance_ratio_.sum():.1%} of variance "
      f"(expected to be low — it's for plotting, not modeling)")

joblib.dump(pca_full, "artifacts/pca_full.joblib")

# 2. K-means — choose k via elbow + silhouette

k_range = range(2, 9)
inertias = []
silhouettes = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].set_title("Elbow method")

axes[1].plot(list(k_range), silhouettes, marker="o", color="#DD8452")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette score by k")
fig.tight_layout()
fig.savefig("plots/kmeans_k_selection.png", dpi=110)
plt.close(fig)

best_k = list(k_range)[int(np.argmax(silhouettes))]
print(f"\nInertia by k: {dict(zip(k_range, np.round(inertias, 1)))}")
print(f"Silhouette by k: {dict(zip(k_range, np.round(silhouettes, 3)))}")
print(f"-> k = {best_k} has the highest silhouette score")

# 3. Fit final k-means

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X)
joblib.dump(kmeans_final, "artifacts/kmeans.joblib")

readable["cluster"] = cluster_labels

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
scatter1 = axes[0].scatter(X_pca2[:, 0], X_pca2[:, 1], c=cluster_labels, cmap="tab10", s=10, alpha=0.7)
axes[0].set_title(f"Customer segments (k={best_k}) in 2D PCA space")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
legend1 = axes[0].legend(*scatter1.legend_elements(), title="Cluster", loc="best", fontsize=8)
axes[0].add_artist(legend1)

colors = np.where(y == 1, "#C44E52", "#4C72B0")
axes[1].scatter(X_pca2[:, 0], X_pca2[:, 1], c=colors, s=10, alpha=0.6)
axes[1].set_title("Actual churn in the same 2D PCA space")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", label="Churned", markersize=7),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", label="Stayed", markersize=7)]
axes[1].legend(handles=handles, loc="best", fontsize=8)
fig.tight_layout()
fig.savefig("plots/pca_clusters_vs_churn.png", dpi=110)
plt.close(fig)

profile_cols = [
    "tenure_months", "satisfaction_score", "monthly_income",
    "number_of_complaints", "customer_service_calls", "late_payment_count",
    "avg_monthly_bill", "app_usage_hours_per_week",
]
profile = readable.groupby("cluster")[profile_cols].mean().round(1)
profile["churn_rate"] = readable.groupby("cluster")["churn_binary"].mean().round(3)
profile["n_customers"] = readable.groupby("cluster").size()
print("\nCluster profiles:")
print(profile.to_string())

profile.to_csv("data/cluster_profiles.csv")
readable.to_csv("data/features_readable_with_clusters.csv", index=False)

print("\nSaved: artifacts/pca_full.joblib, artifacts/kmeans.joblib, "
      "data/cluster_profiles.csv, data/features_readable_with_clusters.csv, plots/*.png")
