import networkx as nx
from awpy.data import NAV, NAV_GRAPHS

map_name = "de_mirage"
g = NAV_GRAPHS[map_name]

print(f"Nodes (nav areas): {g.number_of_nodes()}")
print(f"Edges (adjacencies): {g.number_of_edges()}")
print(f"Is directed: {g.is_directed()}")

print(f"Weakly connected: {nx.is_weakly_connected(g)}")

# Strongly connected: can you reach EVERY node from EVERY other node,
# respecting direction? This is the strict requirement for all-pairs
# shortest path to return a finite distance for every (i, j) pair.
print(f"Strongly connected: {nx.is_strongly_connected(g)}")

if not nx.is_strongly_connected(g):
    components = list(nx.strongly_connected_components(g))
    print(f"Number of strongly connected components: {len(components)}")
    sizes = sorted((len(c) for c in components), reverse=True)
    print(f"Component sizes: {sizes[:10]}{'...' if len(sizes) > 10 else ''}")
