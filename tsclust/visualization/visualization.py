"""
Visualization module for time series clustering results
"""

from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt




def plot_clustering_results(
    X: np.ndarray,
    labels: np.ndarray,
    medoids: np.ndarray,
    title: str = "Time Series Clustering Results",
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot clustering results: time series in each cluster with medoid highlighted

    Parameters
    ----------
    X : np.ndarray
        Time series data with shape (n_samples, series_length)
    labels : np.ndarray
        Sample labels with shape (n_samples,)
    medoids : np.ndarray
        Medoid indices with shape (n_clusters,)
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    save_path : Optional[str]
        If provided, save plot to this path
    """
    n_clusters = len(medoids)
    fig, axes = plt.subplots(n_clusters, 1, figsize=figsize)
    if n_clusters == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        
        # Get all samples in this cluster
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_data = X[cluster_indices]
        
        # Plot all series in cluster (semi-transparent)
        for i, idx in enumerate(cluster_indices):
            ax.plot(
                X[idx],
                alpha=0.3,
                color=colors[cluster_id],
                linewidth=1,
            )
        
        # Highlight medoid
        medoid_idx = medoids[cluster_id]
        ax.plot(
            X[medoid_idx],
            color=colors[cluster_id],
            linewidth=2.5,
            label=f'Medoid (sample {medoid_idx})',
            marker='o',
            markersize=4,
            markevery=max(1, len(X[0]) // 10),
        )
        
        ax.set_title(
            f'Cluster {cluster_id}: {len(cluster_indices)} samples',
            fontsize=12,
            fontweight='bold',
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Value')
    
    axes[-1].set_xlabel('Time Step')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Clustering result plot saved to: {save_path}")
    
    plt.show()


def plot_distance_matrix(
    distance_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Distance Matrix Heatmap",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot distance matrix as heatmap

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix with shape (n_samples, n_samples)
    labels : Optional[np.ndarray]
        Sample labels for reordering matrix by cluster
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : Optional[str]
        If provided, save plot to this path
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reorder by cluster if labels provided
    dist_display = distance_matrix.copy()
    order = np.arange(len(distance_matrix))
    
    if labels is not None:
        order = np.argsort(labels)
        dist_display = distance_matrix[np.ix_(order, order)]
    
    # Plot heatmap
    im = ax.imshow(dist_display, cmap='viridis', aspect='auto')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance', rotation=270, labelpad=15)
    
    # Add cluster boundaries if labels provided
    if labels is not None:
        cluster_boundaries = np.where(np.diff(labels[order]) != 0)[0]
        for boundary in cluster_boundaries:
            ax.axhline(y=boundary + 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=boundary + 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Distance matrix plot saved to: {save_path}")
    
    plt.show()


def plot_medoids_comparison(
    X: np.ndarray,
    labels: np.ndarray,
    medoids: np.ndarray,
    title: str = "Medoids Comparison",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot all medoid series side by side

    Parameters
    ----------
    X : np.ndarray
        Time series data
    labels : np.ndarray
        Sample labels
    medoids : np.ndarray
        Medoid indices
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : Optional[str]
        If provided, save plot to this path
    """
    n_clusters = len(medoids)
    fig, axes = plt.subplots(1, n_clusters, figsize=figsize)
    
    if n_clusters == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster_id, medoid_idx in enumerate(medoids):
        ax = axes[cluster_id]
        
        # Get samples in cluster
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Plot all series in cluster (light background)
        for idx in cluster_indices:
            if idx != medoid_idx:
                ax.plot(X[idx], alpha=0.2, color=colors[cluster_id], linewidth=0.8)
        
        # Highlight medoid
        ax.plot(
            X[medoid_idx],
            color=colors[cluster_id],
            linewidth=2,
            label='Medoid',
            marker='o',
            markersize=3,
            markevery=max(1, len(X[0]) // 8),
        )
        
        ax.set_title(f'Cluster {cluster_id} (n={np.sum(cluster_mask)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.text(0.5, 0.02, 'Time Step', ha='center', fontsize=12)
    axes[0].set_ylabel('Value', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Medoid comparison plot saved to: {save_path}")
    
    plt.show()


def plot_cluster_statistics(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    title: str = "Clustering Statistics",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot clustering statistics: intra-cluster and inter-cluster distances

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix
    labels : np.ndarray
        Sample labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : Optional[str]
        If provided, save plot to this path
    """
    n_clusters = len(np.unique(labels))
    intra_distances = []
    inter_distances = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Intra-cluster distance
        if len(cluster_indices) > 1:
            cluster_dist = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            mask = ~np.eye(len(cluster_indices), dtype=bool)
            intra_distances.append(np.mean(cluster_dist[mask]))
        else:
            intra_distances.append(0)
        
        # Inter-cluster distance
        other_mask = ~cluster_mask
        other_indices = np.where(other_mask)[0]
        if len(other_indices) > 0:
            inter_dist = distance_matrix[np.ix_(cluster_indices, other_indices)]
            inter_distances.append(np.mean(inter_dist))
        else:
            inter_distances.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Intra-cluster distances
    clusters = [f'Cluster {i}' for i in range(n_clusters)]
    ax1.bar(clusters, intra_distances, color='skyblue', alpha=0.7)
    ax1.set_title('Avg Intra-Cluster Distance (lower is better)')
    ax1.set_ylabel('Distance')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Inter-cluster distances
    ax2.bar(clusters, inter_distances, color='lightcoral', alpha=0.7)
    ax2.set_title('Avg Inter-Cluster Distance (higher is better)')
    ax2.set_ylabel('Distance')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Cluster statistics plot saved to: {save_path}")
    
    plt.show()
