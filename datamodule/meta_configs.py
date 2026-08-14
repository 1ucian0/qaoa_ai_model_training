# Copyright IBM Corp. 2026
# License: Apache-2.0

VALID_IN_FEATURES = [
    # Aggregated Graph Features
    "num_nodes",  # Total number of nodes in the loaded graph
    "num_edges",  # Total number of edges in the loaded graph
    "edges_per_node",  # Edge-to-node ratio (num_edges / num_nodes)
    "mean_degree",  # Mean node degree of the graph
    "std_degree",  # Standard deviation of node degrees
    # Raw Graph
    "edges",  # Edge list [[u, v], ...] as a (num_edges, 2) tensor per graph
    "edge_weights",  # Edge weights [w1, w2, ...] as a (num_edges,) tensor per graph
    "t",  # Number of denoising steps for diffusion transformer (training only)
    # Metadata for graph reconstruction (needed for weighted graphs like heavy_hex)
    "node_count",  # Number of nodes, taken from the graph; required by the graph models
    "instance_id",  # Instance identifier for reloading graph with weights
    "instance_set",  # Which instance directory the graph came from
    # === Node Features (per node) ===
    "node_degree",  # Degree of each node
    "node_degree_squared",  # Squared degree per node (energy variance proxy)
    "node_neighbor_degree_mean",  # Mean degree of each node's neighbors
    "node_triangle_count",  # Triangle count per node (3-body correlations)
    "node_clustering",  # Clustering coefficient per node
    "node_betweenness",  # Betweenness centrality per node
    "node_pagerank",  # PageRank score per node
    "node_core_number",  # K-core number per node
    "node_eccentricity",  # Eccentricity per node (max distance to other nodes)
    "node_eigenvector_centrality",  # Eigenvector centrality per node
    # === Alternative Node Features (backward compatibility) ===
    "node_degree_zscore",  # Z-score normalized degree (used in GNN/GIN models)
    "node_triangle_features_combined",  # Combined triangle features (3D: triangles, clustering, density)
    # === Edge Features (per edge) ===
    "edge_triangle_count",  # Number of triangles containing each edge
    "edge_degree_product",  # deg(u) × deg(v) for each edge
    "edge_degree_diff",  # |deg(u) - deg(v)| for each edge
    "edge_degree_min",  # min(deg(u), deg(v)) for each edge
    "edge_degree_max",  # max(deg(u), deg(v)) for each edge
    "edge_jaccard",  # Jaccard coefficient for each edge
    "edge_betweenness",  # Edge betweenness centrality (expensive!)
    "edge_weight",  # Edge weight from graph
    # === Graph-Level Features (scalars) ===
    "graph_diameter",  # Maximum shortest path length
    "graph_avg_path_length",  # Average shortest path length
    "graph_density",  # Edge density
    "graph_clustering",  # Average clustering coefficient
    "graph_transitivity",  # Transitivity (ratio of triangles to triads)
    "graph_assortativity",  # Degree assortativity coefficient
    "graph_num_components",  # Number of connected components
    "graph_algebraic_connectivity",  # Algebraic connectivity (Fiedler value)
    # Rescaling factor from main branch
    "rescale_a",  # Weighing factor for gammas and cost ops based on graph weights
]

VALID_OUT_FEATURES = [
    "qaoa_params",  # Optimized QAOA parameters
    "opt_energy",  # Energy value achieved by the optimized QAOA parameters
    # Max-Cut metrics of optimized QAOA params -> Not computed by default
    "opt_max_cut",
    "opt_min_cut",
    "opt_approx_ratio",  # Approximation ratio between QAOA energy and Max-Cut value
]

# Basic graph features (fast to compute, recommended default)
BASIC_IN_FEATURES = [
    "num_nodes",
    "num_edges",
    "edges_per_node",
    "mean_degree",
    "std_degree",
]

# In Features which are normalized
SCALAR_FEATURES_IN = [
    "num_nodes",
    "num_edges",
    "edges_per_node",
    "mean_degree",
    "std_degree",
    # Graph-level features (scalars) - EXPENSIVE to compute
    "graph_diameter",
    "graph_avg_path_length",
    "graph_density",
    "graph_clustering",
    "graph_transitivity",
    "graph_assortativity",
    "graph_num_components",
    "graph_algebraic_connectivity",
]

# === Feature Groups for Reference ===
# Import these from features.py when needed
NODE_FEATURES = [
    "node_degree",
    "node_degree_squared",
    "node_neighbor_degree_mean",
    "node_triangle_count",
    "node_clustering",
    "node_betweenness",
    "node_pagerank",
    "node_core_number",
    "node_eccentricity",
    "node_eigenvector_centrality",
    "node_degree_zscore",
    "node_triangle_features_combined",
]

EDGE_FEATURES = [
    "edge_triangle_count",
    "edge_degree_product",
    "edge_degree_diff",
    "edge_degree_min",
    "edge_degree_max",
    "edge_jaccard",
    "edge_betweenness",
    "edge_weight",
]

GRAPH_FEATURES = [
    "graph_diameter",
    "graph_avg_path_length",
    "graph_density",
    "graph_clustering",
    "graph_transitivity",
    "graph_assortativity",
    "graph_num_components",
    "graph_algebraic_connectivity",
]
