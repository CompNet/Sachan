## -*- eval: (code-cells-mode); -*-
#
# Experiments with temporal matching
#
# Author: Arthur Amalvy
# 12/06

# %% Setup
from typing import *
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../.."

# TODO: get these from command line
jeffrey_got_repo_path = "~/Dev/game-of-thrones"
gold_alignment_path = "~/Nextcloud/phd/irp/alignment_gold.pickle"


# %%
import glob
import networkx as nx
from ..preprocessing.tvshow.extraction import (
    load_got_tvshow_graphs,
    load_tvshow_character_map,
)


tvshow_charmap = load_tvshow_character_map(f"{root_dir}/in/tvshow/charmap.csv")
tvshow_graphs = load_got_tvshow_graphs(
    jeffrey_got_repo_path, "block", tvshow_charmap, "locations"
)
tvshow_graphs = [
    G for G in tvshow_graphs if G.graph["season"] < 7
]  # ignore season 7 and 8

novel_graphs = []
for path in sorted(glob.glob(f"{root_dir}/in/novels/instant/*.graphml")):
    novel_graphs.append(nx.read_graphml(path))

novel_graphs = [
    nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
    for G in novel_graphs
]


# %%
import copy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from more_itertools import flatten


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
    tvshow_graphs: List[nx.Graph],
    book_graphs: List[nx.Graph],
    mode: Literal["nodes", "edges"],
) -> np.ndarray:
    """
    :return: ``(b, t)``
    """
    M = np.zeros((len(book_graphs), len(tvshow_graphs)))

    # * Keep only common characters
    tvshow_characters = set(flatten([G.nodes for G in tvshow_graphs]))
    book_characters = set(flatten([G.nodes for G in book_graphs]))
    tvshow_xor_book_characters = tvshow_characters ^ book_characters
    tvshow_graphs = [
        filtered_graph(G, list(tvshow_xor_book_characters)) for G in tvshow_graphs
    ]
    book_graphs = [
        filtered_graph(G, list(tvshow_xor_book_characters)) for G in book_graphs
    ]

    # * Nodes mode
    characters_appearances = {char: 0 for char in tvshow_characters & book_characters}
    for G in tvshow_graphs + book_graphs:
        for node in G.nodes:
            characters_appearances[node] += 1

    # * Edges mode
    rel_appearances = {}
    for G in tvshow_graphs + book_graphs:
        for edge in G.edges:
            rel_appearances[tuple(sorted(edge))] = (
                rel_appearances.get(tuple(sorted(edge)), 0) + 1
            )

    # * Compute n^2 similarity
    for chapter_i, chapter_G in enumerate(tqdm(book_graphs)):
        for scene_i, scene_G in enumerate(tvshow_graphs):
            weights = (
                {c: 1 / n for c, n in characters_appearances.items()}
                if mode == "nodes"
                else {c: 1 / n for c, n in rel_appearances.items()}
            )
            M[chapter_i][scene_i] = jaccard_graph_sim(
                chapter_G, scene_G, weights, mode=mode
            )

    return M


# (chapters_nb, blocks_nb)
M_sim = graph_similarity_matrix(tvshow_graphs, novel_graphs, "edges")


def get_episode_i(G: nx.Graph) -> int:
    assert G.graph["season"] < 7
    return (G.graph["season"] - 1) * 10 + G.graph["episode"] - 1


# (blocks_nb)
M_block_to_episode = np.array([get_episode_i(G) for G in tvshow_graphs])


# %%
import pickle


def get_align_matrix(M_sim: np.ndarray, threshold: float) -> np.ndarray:
    """
    :param M_sim: ``(chapters_nb, blocks_nb)``
    :param threshold: between 0 and 1
    :return: ``(episodes_nb, chapters_nb)``
    """

    M_align_blocks = M_sim >= threshold

    _, uniq_start_i = np.unique(M_block_to_episode, return_index=True)
    splits = np.split(M_align_blocks, uniq_start_i[1:], axis=1)

    M_align = []
    for split in splits:
        M_align.append(np.any(split, axis=1))
    return np.stack(M_align)


M_align = get_align_matrix(M_sim, 0.1)

with open(gold_alignment_path, "rb") as f:
    M_align_gold = pickle.load(f)

fig, axs = plt.subplots(2, 1)
axs[0].set_title("Gold alignment")
axs[0].imshow(M_align_gold)
axs[0].set_xlabel("chapters")
axs[0].set_ylabel("episodes")
axs[1].set_title("jaccard-index based alignment")
axs[1].imshow(M_align)
axs[1].set_xlabel("chapters")
axs[1].set_ylabel("episodes")
plt.show()

# %%
from sklearn.metrics import precision_recall_fscore_support

metrics = []
thresholds = np.arange(0, 1, 0.01)
for threshold in thresholds:
    M_align = get_align_matrix(M_sim, threshold)
    precision, recall, f1, _ = precision_recall_fscore_support(
        M_align_gold.flatten(), M_align.flatten(), average="binary"
    )
    metrics.append((precision, recall, f1))

plt.plot(thresholds, [m[0] for m in metrics], label="precision")
plt.plot(thresholds, [m[1] for m in metrics], label="recall")
plt.plot(thresholds, [m[2] for m in metrics], label="f1")
plt.legend()
plt.show()
