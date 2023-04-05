from typing import List, Set, Generator
import copy
import networkx as nx
from more_itertools import flatten


def relabeled_with_id(G: nx.Graph, attribute: str) -> nx.Graph:
    """Relabel a graph with unique number ids as nodes, and the current node as an attribute

    :param G:
    :param attribute:
    """
    H = copy.deepcopy(G)
    for node in H.nodes:
        H.nodes[node][attribute] = node
    H = nx.relabel_nodes(H, {node: i for i, node in enumerate(H.nodes())})
    return H


def graph_edges_attributes(G: nx.Graph) -> Set[str]:
    """Compute the set of all edges attributes of a graph"""
    return set(flatten(list(data.keys()) for *_, data in G.edges.data()))  # type: ignore


def cumulative_graph(graphs: List[nx.Graph]) -> Generator[nx.Graph, None, None]:
    """Turns a dynamic graph to a cumulative graph, weight wise

    :param graphs: A list of sequential graphs
    """
    if len(graphs) == 0:
        return

    all_attrs = set(flatten([graph_edges_attributes(G) for G in graphs]))

    prev_G = graphs[0]
    for H in graphs[1:]:
        # nx.compose creates a new graph with the nodes and edges
        # from both graphs...
        K = nx.compose(H, prev_G)
        # ... however it doesn't sum the attributes : we readjust
        # these here.
        for n1, n2 in K.edges:
            attrs = {}
            for attr in all_attrs:
                G_attr = prev_G.edges.get([n1, n2], default={attr: 0})[attr]
                H_attr = H.edges.get([n1, n2], default={attr: 0})[attr]
                attrs[attr] = G_attr + H_attr
            K.add_edge(n1, n2, **attrs)
        # We also re-add the graph and nodes attributes from G
        K.graph = H.graph
        # yield access to the current cumulative graph
        yield K
        # next iteration
        prev_G = K
