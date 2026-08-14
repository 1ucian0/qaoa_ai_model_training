# Copyright IBM Corp. 2026
# License: Apache-2.0

"""
Compute individual graph, node, and edge features from NetworkX graphs.

This module provides functions to compute various structural features
for nodes, edges, and entire graphs. Features are computed on-the-fly
during dataset loading based on the requested features in the config.
"""

import numpy as np
import networkx as nx
from typing import Optional


# ============================================================================
# Node Feature Computation
# ============================================================================


def compute_node_degree(graph: nx.Graph, num_nodes: int) -> np.ndarray:
    """
    Compute normalised degree for each node.

    Degrees are divided by the maximum degree in the graph so that the
    feature is always in [0, 1], making it comparable across graphs of
    different sizes and densities.

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes

    Returns:
        Array of normalised node degrees (num_nodes,)  in [0, 1]
    """
    degrees = dict(graph.degree())
    raw = np.array([degrees.get(i, 0) for i in range(num_nodes)], dtype=np.float32)
    max_deg = raw.max()
    if max_deg > 0:
        raw = raw / max_deg
    return raw


def compute_node_clustering(graph: nx.Graph, num_nodes: int) -> np.ndarray:
    """
    Compute clustering coefficient for each node.

    The clustering coefficient measures how much a node's neighbors
    are connected to each other (forming triangles).

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes

    Returns:
        Array of clustering coefficients (num_nodes,)
    """
    clustering = nx.clustering(graph)
    return np.array([clustering.get(i, 0) for i in range(num_nodes)], dtype=np.float32)


def compute_node_triangle_count(graph: nx.Graph, num_nodes: int) -> np.ndarray:
    """
    Compute triangle count for each node, normalized by the maximum count.

    Triangle count is the absolute number of triangles a node participates in.
    This differs from clustering coefficient, which is the ratio of actual
    to possible triangles. Triangle count is particularly relevant for QAOA
    as it captures the absolute strength of three-body correlations in the
    cost Hamiltonian.

    Values are divided by the maximum triangle count in the graph so that
    the feature is always in [0, 1], making it comparable across graphs.

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes

    Returns:
        Array of normalized triangle counts (num_nodes,) in [0, 1]
    """
    triangles = nx.triangles(graph)
    raw = np.array([triangles.get(i, 0) for i in range(num_nodes)], dtype=np.float32)
    max_triangles = raw.max()
    if max_triangles > 0:
        raw = raw / max_triangles
    return raw


def compute_node_betweenness(
    graph: nx.Graph, num_nodes: int, k: Optional[int] = None
) -> np.ndarray:
    """
    Compute betweenness centrality for each node.

    Betweenness measures how many shortest paths pass through a node.
    NetworkX already returns values normalised to [0, 1] by default
    (divided by (n-1)(n-2)/2 for undirected graphs).  For large graphs
    (>50 nodes) an approximation with node sampling is used.

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes
        k: Number of nodes to sample for approximation (None = all nodes)

    Returns:
        Array of betweenness centrality values (num_nodes,)  in [0, 1]
    """
    try:
        if k is None and num_nodes > 50:
            k = min(20, num_nodes)
        # Fixed seed so the k-sampled approximation is reproducible across runs
        betweenness = nx.betweenness_centrality(
            graph, k=k, normalized=True, seed=0 if k is not None else None
        )
        return np.array(
            [betweenness.get(i, 0) for i in range(num_nodes)], dtype=np.float32
        )
    except:
        return np.zeros(num_nodes, dtype=np.float32)


def compute_node_pagerank(graph: nx.Graph, num_nodes: int) -> np.ndarray:
    """
    Compute PageRank for each node.

    PageRank measures node importance based on the structure of incoming links.

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes

    Returns:
        Array of PageRank scores (num_nodes,)
    """
    try:
        pagerank = nx.pagerank(graph, max_iter=100)
        return np.array(
            [pagerank.get(i, 1.0 / num_nodes) for i in range(num_nodes)],
            dtype=np.float32,
        )
    except:
        return np.ones(num_nodes, dtype=np.float32) / num_nodes


def compute_node_core_number(graph: nx.Graph, num_nodes: int) -> np.ndarray:
    """
    Compute normalised k-core number for each node.

    The core number is divided by the maximum core number in the graph so
    the feature is always in [0, 1] regardless of graph density.

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes

    Returns:
        Array of normalised core numbers (num_nodes,)  in [0, 1]
    """
    try:
        core_number = nx.core_number(graph)
        raw = np.array(
            [core_number.get(i, 0) for i in range(num_nodes)], dtype=np.float32
        )
        max_core = raw.max()
        if max_core > 0:
            raw = raw / max_core
        return raw
    except:
        return np.zeros(num_nodes, dtype=np.float32)


def compute_node_eccentricity(graph: nx.Graph, num_nodes: int) -> np.ndarray:
    """
    Compute normalised eccentricity for each node.

    Eccentricity is the maximum shortest-path distance from a node to any
    other node, divided by the graph diameter so the feature is in [0, 1].
    Only computed for small (<100 nodes) connected graphs due to cost.

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes

    Returns:
        Array of normalised eccentricities (num_nodes,)  in [0, 1]
    """
    try:
        if nx.is_connected(graph) and num_nodes < 100:
            eccentricity = nx.eccentricity(graph)
            raw = np.array(
                [eccentricity.get(i, 0) for i in range(num_nodes)], dtype=np.float32
            )
            diam = raw.max()
            if diam > 0:
                raw = raw / diam
            return raw
        else:
            return np.zeros(num_nodes, dtype=np.float32)
    except:
        return np.zeros(num_nodes, dtype=np.float32)


def compute_node_eigenvector_centrality(graph: nx.Graph, num_nodes: int) -> np.ndarray:
    """
    Compute eigenvector centrality for each node.

    Eigenvector centrality measures node importance based on connections
    to other important nodes.

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes

    Returns:
        Array of eigenvector centrality values (num_nodes,)
    """
    try:
        centrality = nx.eigenvector_centrality(graph, max_iter=100)
        return np.array(
            [centrality.get(i, 0) for i in range(num_nodes)], dtype=np.float32
        )
    except:
        return np.zeros(num_nodes, dtype=np.float32)


def compute_node_degree_squared(graph: nx.Graph, num_nodes: int) -> np.ndarray:
    """
    Compute normalised squared degree for each node.

    The squared degree captures the nonlinear relationship between local
    connectivity and QAOA cost-function energy statistics.  In QAOA the
    second-order energy variance (which controls the landscape curvature
    around optimal gamma) scales with degree², making this a
    physics-informed feature for parameter prediction.

    Values are divided by the maximum squared degree in the graph so that
    the feature is always in [0, 1].

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes

    Returns:
        Array of normalised squared degrees (num_nodes,)  in [0, 1]
    """
    degrees = dict(graph.degree())
    raw = np.array([degrees.get(i, 0) for i in range(num_nodes)], dtype=np.float32)
    raw = raw**2
    max_val = raw.max()
    if max_val > 0:
        raw = raw / max_val
    return raw


def compute_node_neighbor_degree_mean(graph: nx.Graph, num_nodes: int) -> np.ndarray:
    """
    Compute normalised mean neighbor degree for each node.

    For each node *i* the feature is the average degree of *i*'s
    neighbors:  d_N(i) = (1/|N(i)|) * sum_{j in N(i)} deg(j).
    Isolated nodes (degree 0) receive a value of 0.

    In QAOA the optimal gamma for a qubit depends not only on its own
    degree but also on the degrees of the qubits it shares ZZ
    interactions with.  This "neighbor-degree" signal is a key
    structural feature that governs parameter concentration and
    transferability across graphs (Brandao et al., 2018; Wurtz & Love,
    2021).

    Values are divided by the maximum mean neighbor degree in the graph
    so the feature is always in [0, 1].

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes

    Returns:
        Array of normalised mean neighbor degrees (num_nodes,)  in [0, 1]
    """
    degrees = dict(graph.degree())
    raw = np.zeros(num_nodes, dtype=np.float32)
    for i in range(num_nodes):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) > 0:
            raw[i] = np.mean([degrees.get(j, 0) for j in neighbors])
    max_val = raw.max()
    if max_val > 0:
        raw = raw / max_val
    return raw


# ============================================================================
# Edge Feature Computation
# ============================================================================


def _is_padding(u: int, v: int, graph: nx.Graph) -> bool:
    """
    Return True if (u, v) is a padding entry, i.e. the edge does not exist in
    the graph.  We cannot use ``u == 0 and v == 0`` as a sentinel because node
    0 can genuinely participate in real edges.
    """
    return not graph.has_edge(u, v)


def compute_edge_triangle_count(graph: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Count triangles containing each edge, normalised by the maximum triangle
    count across all edges so the feature is in [0, 1].

    A triangle is formed when both endpoints of an edge share a common neighbor.

    Args:
        graph: NetworkX graph
        edge_list: List of edges (potentially padded)

    Returns:
        Array of normalised triangle counts per edge (num_edges,)  in [0, 1]
    """
    features = []
    for u, v in edge_list:
        if _is_padding(u, v, graph):
            features.append(0.0)
        else:
            try:
                neighbors_u = set(graph.neighbors(u))
                neighbors_v = set(graph.neighbors(v))
                triangles = len(neighbors_u.intersection(neighbors_v))
                features.append(float(triangles))
            except:
                features.append(0.0)
    raw = np.array(features, dtype=np.float32)
    max_val = raw.max()
    if max_val > 0:
        raw = raw / max_val
    return raw


def compute_edge_degree_product(graph: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Compute degree(u) × degree(v) for each edge, normalised by the maximum
    product in the graph so the feature is in [0, 1].

    Higher values indicate edges connecting high-degree nodes.

    Args:
        graph: NetworkX graph
        edge_list: List of edges

    Returns:
        Array of normalised degree products (num_edges,)  in [0, 1]
    """
    features = []
    for u, v in edge_list:
        if _is_padding(u, v, graph):
            features.append(0.0)
        else:
            try:
                features.append(float(graph.degree(u) * graph.degree(v)))
            except:
                features.append(0.0)
    raw = np.array(features, dtype=np.float32)
    max_val = raw.max()
    if max_val > 0:
        raw = raw / max_val
    return raw


def compute_edge_degree_diff(graph: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Compute |degree(u) - degree(v)| for each edge, normalised by the maximum
    degree in the graph so the feature is in [0, 1].

    Measures degree asymmetry between edge endpoints.

    Args:
        graph: NetworkX graph
        edge_list: List of edges

    Returns:
        Array of normalised degree differences (num_edges,)  in [0, 1]
    """
    max_deg = (
        float(max(d for _, d in graph.degree())) if graph.number_of_nodes() > 0 else 1.0
    )
    features = []
    for u, v in edge_list:
        if _is_padding(u, v, graph):
            features.append(0.0)
        else:
            try:
                features.append(abs(graph.degree(u) - graph.degree(v)) / max_deg)
            except:
                features.append(0.0)
    return np.array(features, dtype=np.float32)


def compute_edge_degree_min(graph: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Compute min(degree(u), degree(v)) for each edge, normalised by the maximum
    degree in the graph so the feature is in [0, 1].

    Args:
        graph: NetworkX graph
        edge_list: List of edges

    Returns:
        Array of normalised minimum endpoint degrees (num_edges,)  in [0, 1]
    """
    max_deg = (
        float(max(d for _, d in graph.degree())) if graph.number_of_nodes() > 0 else 1.0
    )
    features = []
    for u, v in edge_list:
        if _is_padding(u, v, graph):
            features.append(0.0)
        else:
            try:
                features.append(min(graph.degree(u), graph.degree(v)) / max_deg)
            except:
                features.append(0.0)
    return np.array(features, dtype=np.float32)


def compute_edge_degree_max(graph: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Compute max(degree(u), degree(v)) for each edge, normalised by the maximum
    degree in the graph so the feature is in [0, 1].

    Args:
        graph: NetworkX graph
        edge_list: List of edges

    Returns:
        Array of normalised maximum endpoint degrees (num_edges,)  in [0, 1]
    """
    max_deg = (
        float(max(d for _, d in graph.degree())) if graph.number_of_nodes() > 0 else 1.0
    )
    features = []
    for u, v in edge_list:
        if _is_padding(u, v, graph):
            features.append(0.0)
        else:
            try:
                features.append(max(graph.degree(u), graph.degree(v)) / max_deg)
            except:
                features.append(0.0)
    return np.array(features, dtype=np.float32)


def compute_edge_jaccard(graph: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Compute Jaccard coefficient for each edge.

    Jaccard = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|,  already in [0, 1].

    Args:
        graph: NetworkX graph
        edge_list: List of edges

    Returns:
        Array of Jaccard coefficients (num_edges,)  in [0, 1]
    """
    features = []
    for u, v in edge_list:
        if _is_padding(u, v, graph):
            features.append(0.0)
        else:
            try:
                neighbors_u = set(graph.neighbors(u))
                neighbors_v = set(graph.neighbors(v))
                intersection = len(neighbors_u.intersection(neighbors_v))
                union = len(neighbors_u.union(neighbors_v))
                features.append(float(intersection / union) if union > 0 else 0.0)
            except:
                features.append(0.0)
    return np.array(features, dtype=np.float32)


def compute_edge_betweenness(graph: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Compute edge betweenness centrality (normalised to [0, 1] by NetworkX).

    Measures how many shortest paths pass through an edge.
    WARNING: This is computationally expensive for large graphs!

    Args:
        graph: NetworkX graph
        edge_list: List of edges

    Returns:
        Array of edge betweenness values (num_edges,)  in [0, 1]
    """
    try:
        betweenness = nx.edge_betweenness_centrality(graph, normalized=True)
        features = []
        for u, v in edge_list:
            if _is_padding(u, v, graph):
                features.append(0.0)
            else:
                # NetworkX stores edges as (u, v) or (v, u)
                val = betweenness.get((u, v), betweenness.get((v, u), 0.0))
                features.append(float(val))
        return np.array(features, dtype=np.float32)
    except:
        return np.zeros(len(edge_list), dtype=np.float32)


def compute_edge_weight(graph: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Extract edge weights from graph.

    Args:
        graph: NetworkX graph
        edge_list: List of edges

    Returns:
        Array of edge weights (num_edges,)
    """
    features = []
    for u, v in edge_list:
        if _is_padding(u, v, graph):
            features.append(0.0)
        else:
            try:
                features.append(float(graph[u][v].get("weight", 1.0)))
            except:
                features.append(1.0)
    return np.array(features, dtype=np.float32)


# ============================================================================
# Graph-Level Feature Computation
# ============================================================================


def compute_graph_diameter(graph: nx.Graph) -> float:
    """
    Compute graph diameter (maximum shortest path length).

    Only computed for small (<100 nodes) connected graphs.

    Args:
        graph: NetworkX graph

    Returns:
        Diameter value
    """
    try:
        if nx.is_connected(graph) and graph.number_of_nodes() < 100:
            return float(nx.diameter(graph))
        else:
            return 0.0
    except:
        return 0.0


def compute_graph_avg_path_length(graph: nx.Graph) -> float:
    """
    Compute average shortest path length.

    Only computed for small (<100 nodes) connected graphs.

    Args:
        graph: NetworkX graph

    Returns:
        Average path length
    """
    try:
        if nx.is_connected(graph) and graph.number_of_nodes() < 100:
            return float(nx.average_shortest_path_length(graph))
        else:
            return 0.0
    except:
        return 0.0


def compute_graph_density(graph: nx.Graph) -> float:
    """
    Compute graph density (ratio of actual edges to possible edges).

    Args:
        graph: NetworkX graph

    Returns:
        Density value in [0, 1]
    """
    try:
        return float(nx.density(graph))
    except:
        return 0.0


def compute_graph_clustering(graph: nx.Graph) -> float:
    """
    Compute average clustering coefficient.

    Args:
        graph: NetworkX graph

    Returns:
        Average clustering coefficient
    """
    try:
        return float(nx.average_clustering(graph))
    except:
        return 0.0


def compute_graph_transitivity(graph: nx.Graph) -> float:
    """
    Compute transitivity (ratio of triangles to connected triples).

    Args:
        graph: NetworkX graph

    Returns:
        Transitivity value
    """
    try:
        return float(nx.transitivity(graph))
    except:
        return 0.0


def compute_graph_assortativity(graph: nx.Graph) -> float:
    """
    Compute degree assortativity coefficient.

    Measures tendency of nodes to connect to others with similar degree.

    Args:
        graph: NetworkX graph

    Returns:
        Assortativity coefficient in [-1, 1]
    """
    try:
        result = nx.degree_assortativity_coefficient(graph)
        # Handle NaN values (can occur for regular graphs or other special cases)
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return float(result)
    except:
        return 0.0


def compute_graph_num_components(graph: nx.Graph) -> float:
    """
    Count number of connected components.

    Args:
        graph: NetworkX graph

    Returns:
        Number of connected components
    """
    try:
        return float(nx.number_connected_components(graph))
    except:
        return 1.0


def compute_graph_algebraic_connectivity(graph: nx.Graph) -> float:
    """
    Compute algebraic connectivity (second smallest eigenvalue of Laplacian).

    Measures how well connected the graph is. Only computed for small
    (<100 nodes) connected graphs.

    Args:
        graph: NetworkX graph

    Returns:
        Algebraic connectivity value
    """
    try:
        if nx.is_connected(graph) and graph.number_of_nodes() < 100:
            return float(nx.algebraic_connectivity(graph))
        else:
            return 0.0
    except:
        return 0.0


# ============================================================================
# Alternative Feature Computation (from model implementations)
# ============================================================================
# These functions preserve the exact computation logic from the models
# for backward compatibility and comparison purposes.
#
# USAGE NOTES:
# - node_degree_zscore: Requires mean_degree and std_degree parameters.
#   In dataset.py, these are computed from the graph and passed explicitly.
#   Signature: compute_node_degree_zscore(graph, num_nodes, mean_deg, std_deg)
#
# - node_triangle_features_combined: Returns (num_nodes, 3) array instead of
#   (num_nodes,) like standard node features. The 3 features are:
#   [0] Triangle count per node (z-score normalized)
#   [1] Clustering coefficient per node
#   [2] Global triangle density (same for all nodes)
#   Signature: compute_triangle_features_combined(graph, num_nodes)


def compute_node_degree_zscore(
    graph: nx.Graph, num_nodes: int, mean_deg: float, std_deg: float
) -> np.ndarray:
    """
    Compute z-score normalized degree for each node.

    This matches the normalization used in GNN/GIN models:
    deg_norm = (deg - mean_deg) / std_deg

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes
        mean_deg: Mean degree for normalization
        std_deg: Standard deviation of degrees for normalization

    Returns:
        Array of z-score normalized node degrees (num_nodes,)
    """
    degrees = dict(graph.degree())
    raw = np.array([degrees.get(i, 0) for i in range(num_nodes)], dtype=np.float32)

    std_safe = max(std_deg, 1e-6)
    deg_norm = (raw - mean_deg) / std_safe

    return deg_norm


def compute_triangle_features_combined(graph: nx.Graph, num_nodes: int) -> np.ndarray:
    """
    Compute combined triangle-based features as used in GIN model.

    Returns a 3-feature vector per node:
    - Column 0: Triangle count per node (z-score normalized)
    - Column 1: Clustering coefficient per node
    - Column 2: Global triangle density (same for all nodes)

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes

    Returns:
        Array of triangle features (num_nodes, 3)
    """
    # Triangle count per node
    triangles_dict = nx.triangles(graph)
    triangles_per_node = np.array(
        [triangles_dict.get(i, 0) for i in range(num_nodes)], dtype=np.float32
    )

    # Clustering coefficient per node
    clustering_dict = nx.clustering(graph)
    clustering_per_node = np.array(
        [clustering_dict.get(i, 0) for i in range(num_nodes)], dtype=np.float32
    )

    # Normalize triangle counts (z-score)
    mean_triangles = triangles_per_node.mean()
    std_triangles = triangles_per_node.std()
    if std_triangles > 1e-6:
        normalized_triangles = (triangles_per_node - mean_triangles) / std_triangles
    else:
        normalized_triangles = triangles_per_node - mean_triangles

    # Global triangle density
    total_triangles = triangles_per_node.sum() / 3.0
    max_triangles = num_nodes * (num_nodes - 1) * (num_nodes - 2) / 6.0
    triangle_density = total_triangles / (max_triangles + 1e-10)

    # Stack features
    triangle_features = np.stack(
        [
            normalized_triangles,
            clustering_per_node,
            np.full(num_nodes, triangle_density, dtype=np.float32),
        ],
        axis=-1,
    )

    return triangle_features


# ============================================================================
# Feature Dispatcher
# ============================================================================

# Map feature names to computation functions
NODE_FEATURE_FUNCTIONS = {
    "node_degree": compute_node_degree,
    "node_degree_squared": compute_node_degree_squared,
    "node_neighbor_degree_mean": compute_node_neighbor_degree_mean,
    "node_triangle_count": compute_node_triangle_count,
    "node_clustering": compute_node_clustering,
    "node_betweenness": compute_node_betweenness,
    "node_pagerank": compute_node_pagerank,
    "node_core_number": compute_node_core_number,
    "node_eccentricity": compute_node_eccentricity,
    "node_eigenvector_centrality": compute_node_eigenvector_centrality,
    "node_degree_zscore": compute_node_degree_zscore,
    "node_triangle_features_combined": compute_triangle_features_combined,
}

EDGE_FEATURE_FUNCTIONS = {
    "edge_triangle_count": compute_edge_triangle_count,
    "edge_degree_product": compute_edge_degree_product,
    "edge_degree_diff": compute_edge_degree_diff,
    "edge_degree_min": compute_edge_degree_min,
    "edge_degree_max": compute_edge_degree_max,
    "edge_jaccard": compute_edge_jaccard,
    "edge_betweenness": compute_edge_betweenness,
    "edge_weight": compute_edge_weight,
}

GRAPH_FEATURE_FUNCTIONS = {
    "graph_diameter": compute_graph_diameter,
    "graph_avg_path_length": compute_graph_avg_path_length,
    "graph_density": compute_graph_density,
    "graph_clustering": compute_graph_clustering,
    "graph_transitivity": compute_graph_transitivity,
    "graph_assortativity": compute_graph_assortativity,
    "graph_num_components": compute_graph_num_components,
    "graph_algebraic_connectivity": compute_graph_algebraic_connectivity,
}

# Feature groups for easier reference
NODE_FEATURES = list(NODE_FEATURE_FUNCTIONS.keys())
EDGE_FEATURES = list(EDGE_FEATURE_FUNCTIONS.keys())
GRAPH_FEATURES = list(GRAPH_FEATURE_FUNCTIONS.keys())
