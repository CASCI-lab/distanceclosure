import networkx as nx
from distanceclosure import distance_backbone

def testing_module_undirected() -> dict[str, nx.Graph | str]: 
    """
    Hand constructed and validated undirected 4 node weighted graph.
    """

    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, 1),
        (0, 2, 3),
        (0, 3, 3),
        (1, 2, 1),
        (1, 3, 1),
        (2, 3, 2)
    ])
    
    METRIC_KEY = nx.Graph()
    METRIC_KEY.add_weighted_edges_from([
        (0, 1, 1),
        (1, 2, 1),
        (1, 3, 1),
        (2, 3, 2)
    ])
    
    ULTRAMETRIC_KEY = nx.Graph()
    ULTRAMETRIC_KEY.add_weighted_edges_from([
        (0, 1, 1),
        (1, 2, 1),
        (1, 3, 1)
    ])
    
    return {
        "title": "undirected",
        "graph": G,
        "metric_key": METRIC_KEY,
        "ultrametric_key": ULTRAMETRIC_KEY
    }

def testing_module_directed() -> dict[str, nx.DiGraph | str]: 
    """
    Hand constructed and validated directed 4 node weighted graph.
    """

    G = nx.DiGraph()
    G.add_weighted_edges_from([
        (0, 1, 1),
        (0, 2, 3),
        (0, 3, 3),
        (1, 2, 1),
        (1, 3, 1),
        (2, 3, 2)
    ])
    
    METRIC_KEY = nx.DiGraph()
    METRIC_KEY.add_weighted_edges_from([
        (0, 1, 1),
        (1, 2, 1),
        (1, 3, 1),
        (2, 3, 2)
    ]) 

    ULTRAMETRIC_KEY = nx.DiGraph()
    ULTRAMETRIC_KEY.add_weighted_edges_from([
        (0, 1, 1),
        (1, 2, 1),
        (1, 3, 1),
        (2, 3, 2)
    ]) 
    
    return {
        "title": "directed",
        "graph": G,
        "metric_key": METRIC_KEY,
        "ultrametric_key": ULTRAMETRIC_KEY 
    }
    
def testing_module_large_weights() -> dict[str, nx.Graph | str]: 
    """
    Large weights.
    """
    scale = 1e100
    
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, (1 * scale)),
        (0, 2, (3 * scale)),
        (0, 3, (3 * scale)),
        (1, 2, (1 * scale)),
        (1, 3, (1 * scale)),
        (2, 3, (2 * scale))
    ])
    
    METRIC_KEY = nx.Graph()
    METRIC_KEY.add_weighted_edges_from([
        (0, 1, (1 * scale)),
        (1, 2, (1 * scale)),
        (1, 3, (1 * scale)),
        (2, 3, (2 * scale))
    ])
    
    ULTRAMETRIC_KEY = nx.Graph()
    ULTRAMETRIC_KEY.add_weighted_edges_from([
        (0, 1, (1 * scale)),
        (1, 2, (1 * scale)),
        (1, 3, (1 * scale))
    ])
    
    return {
        "title": "large weights",
        "graph": G,
        "metric_key": METRIC_KEY,
        "ultrametric_key": ULTRAMETRIC_KEY
    }
    
def testing_module_small_weights() -> dict[str, nx.Graph | str]: 
    """
    Small weights.
    """
    scale = 1e-100
    
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, (1 * scale)),
        (0, 2, (3 * scale)),
        (0, 3, (3 * scale)),
        (1, 2, (1 * scale)),
        (1, 3, (1 * scale)),
        (2, 3, (2 * scale))
    ])
    
    METRIC_KEY = nx.Graph()
    METRIC_KEY.add_weighted_edges_from([
        (0, 1, (1 * scale)),
        (1, 2, (1 * scale)),
        (1, 3, (1 * scale)),
        (2, 3, (2 * scale))
    ])
    
    ULTRAMETRIC_KEY = nx.Graph()
    ULTRAMETRIC_KEY.add_weighted_edges_from([
        (0, 1, (1 * scale)),
        (1, 2, (1 * scale)),
        (1, 3, (1 * scale))
    ])
    
    return {
        "title": "small weights",
        "graph": G,
        "metric_key": METRIC_KEY,
        "ultrametric_key": ULTRAMETRIC_KEY
    }

def testing_module_no_weights() -> dict[str, nx.Graph | str]: 
    """
    Hand constructed and validated undirected 4 node weighted graph.
    """

    G = nx.Graph()
    G.add_edges_from([
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3)
    ])
    
    METRIC_KEY = G.copy()
    ULTRAMETRIC_KEY = G.copy() 
    
    return {
        "title": "no weights",
        "graph": G,
        "metric_key": METRIC_KEY,
        "ultrametric_key": ULTRAMETRIC_KEY
    }

def testing_module_disconnected() -> dict[str, nx.Graph | str]: 
    """
    Disconnected graph.
    """

    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, 1),
        (2, 3, 2)
    ])
    
    METRIC_KEY = G.copy()
    ULTRAMETRIC_KEY = G.copy()
    
    return {
        "title": "disconnected",
        "graph": G,
        "metric_key": METRIC_KEY,
        "ultrametric_key": ULTRAMETRIC_KEY
    }
    
def testing_module_single_node() -> dict[str, nx.Graph | str]: 
    """
    Single node graph.
    """

    G = nx.empty_graph(1)
    METRIC_KEY = G.copy() 
    ULTRAMETRIC_KEY = G.copy() 
    
    return {
        "title": "single node",
        "graph": G,
        "metric_key": METRIC_KEY,
        "ultrametric_key": ULTRAMETRIC_KEY 
    }

def testing_module_single_edge() -> dict[str, nx.Graph | str]: 
    """
    Single edge graph.
    """

    G = nx.Graph()
    G.add_edge(0, 1, weight=1)

    METRIC_KEY = G.copy()
    ULTRAMETRIC_KEY = G.copy() 
    
    return {
        "title": "single edge",
        "graph": G,
        "metric_key": METRIC_KEY,
        "ultrametric_key": ULTRAMETRIC_KEY 
    }

def assert_graph(graph_to_test: nx.Graph | nx.DiGraph, answer_key: nx.Graph | nx.DiGraph) -> None:
    assert set(graph_to_test.nodes()) == set(answer_key.nodes())
    assert set(graph_to_test.edges()) == set(answer_key.edges())

def test_backbone(module: dict[str, nx.Graph | str]) -> None:
    algorithms_to_test = [
        "ttlp",
        "stlp",
        "ksl",
        "closure",
    ]

    geometry_to_test = [
        "metric",
        "ultrametric"
    ]
    
    for geometry in geometry_to_test:
        for algorithm in algorithms_to_test:
            backbone_to_test = distance_backbone(module["graph"], geometry=geometry, algorithm=algorithm)
            backbone_answer_key = module["metric_key"] if geometry == "metric" else module["ultrametric_key"]
            
            assert_graph(backbone_to_test, backbone_answer_key)
            print(f"[PASS] {module['title'].upper()} {geometry.upper()} {algorithm.upper()}")
            
 
testing_modules = [
    testing_module_undirected,
    testing_module_directed,
    testing_module_single_edge,
    testing_module_disconnected,
    testing_module_large_weights,
    testing_module_small_weights,
    testing_module_no_weights,
    testing_module_single_node,
]

for module in testing_modules:
    test_backbone(module=module())
 
