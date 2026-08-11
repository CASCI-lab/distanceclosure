from distanceclosure import heuristic_backbone, backbone_from_closure
import networkx as nx

path_to_graph = ""
G = nx.read_graphml(path_to_graph)

kinds = [
    "metric",
    "ultrametric"
]

control = G.copy()

for kind in kinds:
    heuristic = heuristic_backbone(G, weight="distance", kind=kind)
    heuristic_approximation = heuristic_backbone(G, weight="distance", kind=kind, approx=True)
    actual_backbone = backbone_from_closure(G, weight="distance", kind=kind)

    print(f"Heurisitc {kind} backbone: {heuristic}")
    print(f"Heurisitc Approximation of {kind} backbone: {heuristic_approximation}")
    print(f"Actual {kind} backbone: {actual_backbone}")
    print(f"Original Graph: {control}")
    print()

