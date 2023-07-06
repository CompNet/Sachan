# Temporal matching between the novels and the comics
#
# Example usage: python exp_temporal_match_novels_comics.py -g ./novels_comics_gold_alignment.pickle
#
# Author: Arthur Amalvy
from typing import List, Literal, Optional, Tuple
import os, sys, glob, copy, argparse, pickle
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from more_itertools import flatten


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")


def filtered_graph(G: nx.Graph, nodes: list) -> nx.Graph:
    H = copy.deepcopy(G)
    for node in nodes:
        if node in H:
            H.remove_node(node)
    return H


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
    comics_graphs: List[nx.Graph],
    book_graphs: List[nx.Graph],
    mode: Literal["nodes", "edges"],
    use_weights: bool,
) -> np.ndarray:
    """
    :return: ``(b, t)``
    """
    M = np.zeros((len(book_graphs), len(comics_graphs)))

    # * Keep only common characters
    tvshow_characters = set(flatten([G.nodes for G in comics_graphs]))
    book_characters = set(flatten([G.nodes for G in book_graphs]))
    tvshow_xor_book_characters = tvshow_characters ^ book_characters
    comics_graphs = [
        filtered_graph(G, list(tvshow_xor_book_characters)) for G in comics_graphs
    ]
    book_graphs = [
        filtered_graph(G, list(tvshow_xor_book_characters)) for G in book_graphs
    ]

    # * Nodes mode
    characters_appearances = {char: 0 for char in tvshow_characters & book_characters}
    for G in comics_graphs + book_graphs:
        for node in G.nodes:
            characters_appearances[node] += 1

    # * Edges mode
    rel_appearances = {}
    for G in comics_graphs + book_graphs:
        for edge in G.edges:
            rel_appearances[tuple(sorted(edge))] = (
                rel_appearances.get(tuple(sorted(edge)), 0) + 1
            )

    # * Compute n^2 similarity
    for chapter_i, chapter_G in enumerate(tqdm(book_graphs)):
        for scene_i, scene_G in enumerate(comics_graphs):
            weights = None
            if use_weights:
                weights = (
                    {c: 1 / n for c, n in characters_appearances.items()}
                    if mode == "nodes"
                    else {c: 1 / n for c, n in rel_appearances.items()}
                )
            M[chapter_i][scene_i] = jaccard_graph_sim(
                chapter_G, scene_G, weights, mode=mode
            )

    return M


if __name__ == "__main__":

    FONTSIZE = 12

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gold-alignment-path", type=str)
    args = parser.parse_args()

    novels_graphs = []
    for path in sorted(glob.glob(f"{root_dir}/in/novels/instant/1.AGoT*.graphml")):
        novels_graphs.append(nx.read_graphml(path))
    for path in sorted(glob.glob(f"{root_dir}/in/novels/instant/2.ACoK*.graphml")):
        novels_graphs.append(nx.read_graphml(path))
    novels_graphs = [
        nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
        for G in novels_graphs
    ]
    assert len(novels_graphs) > 0

    comics_graphs = []
    for path in sorted(glob.glob(f"{root_dir}/in/comics/instant/chapter/*.graphml")):
        comics_graphs.append(nx.read_graphml(path))
    # TODO: check
    comics_graphs = [
        nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
        for G in comics_graphs
    ]
    assert len(comics_graphs) > 0

    # TODO: compute gold alignment
    with open(args.gold_alignment_path, "rb") as f:
        # (novels_chapters_nb, comics_chapters_nb)
        M_align_gold = pickle.load(f)

    # (novels_chapters_nb, comics_chapters_nb)
    M_sim = graph_similarity_matrix(
        comics_graphs, novels_graphs, "edges", use_weights=True
    )

    # Compute (precision, recall, F1) and the best threshold
    metrics: List[Tuple[float, float, float]] = []
    thresholds = np.arange(0, 1, 0.01)
    best_f1 = 0.0
    best_threshold = 0.0
    best_M_align = M_sim >= 0.0
    for threshold in thresholds:
        M_align = M_sim > threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            M_align_gold.flatten(),
            M_align.flatten(),
            average="binary",
            zero_division=0.0,
        )
        metrics.append((precision, recall, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_M_align = M_align
            best_threshold = threshold

    # Plot alignment in a first figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].imshow(M_align_gold)
    axs[0].set_title("Gold alignment", fontsize=FONTSIZE)
    axs[0].set_xlabel("chapters", fontsize=FONTSIZE)
    axs[0].set_ylabel("comics", fontsize=FONTSIZE)
    axs[1].imshow(best_M_align)
    axs[1].set_title(
        f"Jaccard similarity alignment (threshold: {best_threshold})", fontsize=FONTSIZE
    )
    axs[1].set_xlabel("chapters", fontsize=FONTSIZE)
    axs[1].set_ylabel("comics", fontsize=FONTSIZE)

    # Plot precision, recall and F1 in a second figure
    plt.figure(2, figsize=(16, 10))
    plt.plot(thresholds, [m[0] for m in metrics], label="precision")
    plt.plot(thresholds, [m[1] for m in metrics], label="recall")
    plt.plot(thresholds, [m[2] for m in metrics], label="f1")
    plt.xlabel("Chapters similarity threshold", fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)

    # Show both figures
    plt.show()
