# Common functions for temporal matching
#
# Author: Arthur Amalvy
from typing import Optional, Literal, List
import copy
import numpy as np
import networkx as nx
from more_itertools import flatten
from tqdm import tqdm


def filtered_graph(G: nx.Graph, nodes: list) -> nx.Graph:
    """Return a graph where ``nodes`` are removed from ``G``"""
    H = copy.deepcopy(G)
    for node in nodes:
        if node in H:
            H.remove_node(node)
    return H


def get_episode_i(G: nx.Graph) -> int:
    """Get the index of the episode of ``G``

    .. note::

        only supports seasons from 1 to 6
    """
    assert G.graph["season"] < 7
    return (G.graph["season"] - 1) * 10 + G.graph["episode"] - 1


def jaccard_graph_sim(
    G: nx.Graph, H: nx.Graph, weights: Optional[dict], mode: Literal["nodes", "edges"]
) -> float:
    if mode == "nodes":
        g_set = set(G.nodes)
        h_set = set(H.nodes)
    elif mode == "edges":
        g_set = set([tuple(sorted(e)) for e in G.edges])
        h_set = set([tuple(sorted(e)) for e in H.edges])

    if not weights is None:
        inter = sum([weights.get(n, 0) for n in g_set.intersection(h_set)])
        union = sum([weights.get(n, 0) for n in g_set.union(h_set)])
    else:
        inter = len(g_set.intersection(h_set))
        union = len(g_set.union(h_set))

    if union == 0:
        return 0
    return inter / union


def graph_similarity_matrix(
    G_lst: List[nx.Graph],
    H_lst: List[nx.Graph],
    mode: Literal["nodes", "edges"],
    use_weights: bool,
) -> np.ndarray:
    """Compute a similarity matrixs between two lists of graph, using
    a tweaked Jaccard index.

    :param G_lst: a list of graphs of len ``n``
    :param H_lst: a list of graphs of len ``m``
    :param mode: Jaccard index compute methods

    :param use_weights: wether or not to weight Jaccard index by the
        number of occurences of:

            - Each character, if ``mode == "nodes"``

            - Interactions between characters, if ``mode == "edges"``

    :return: a similarity matrix of shape ``(n, m)``
    """
    # Keep only common characters
    G_chars = set(flatten([G.nodes for G in G_lst]))
    H_chars = set(flatten([H.nodes for H in H_lst]))
    G_xor_H_chars = G_chars ^ H_chars
    G_lst = [filtered_graph(G, list(G_xor_H_chars)) for G in G_lst]
    H_lst = [filtered_graph(H, list(G_xor_H_chars)) for H in H_lst]

    # Nodes mode
    char_appearances = {char: 0 for char in G_chars & H_chars}
    for G in G_lst + H_lst:
        for node in G.nodes:
            char_appearances[node] += 1

    # Edges mode
    rel_appearances = {}
    for G in G_lst + H_lst:
        for edge in G.edges:
            rel_appearances[tuple(sorted(edge))] = (
                rel_appearances.get(tuple(sorted(edge)), 0) + 1
            )

    # Compute n^2 similarity
    M = np.zeros((len(G_lst), len(H_lst)))

    for G_i, G in enumerate(tqdm(G_lst)):
        for H_i, H in enumerate(H_lst):
            weights = None
            if use_weights:
                weights = (
                    {c: 1 / n for c, n in char_appearances.items()}
                    if mode == "nodes"
                    else {c: 1 / n for c, n in rel_appearances.items()}
                )
            M[G_i][H_i] = jaccard_graph_sim(G, H, weights, mode=mode)

    return M
