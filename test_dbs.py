from distanceclosure.edge_measures import dombi_synthesis
import networkx as nx

weighted_edges = [(1, 2, 0.5), (2, 3, 0.5), (3, 4, 0.25), (2, 4, 1/15),
                  (4, 5, 0.2), (5, 1, 0.2), (1, 3, 0.2), (3, 5, 1./6)]

G = nx.Graph()
G.add_weighted_edges_from(weighted_edges)

G = dombi_synthesis(G, prox_weight='weight', L=1e-2, R=1e2)
gdf = nx.to_pandas_edgelist(G)

print(gdf)
