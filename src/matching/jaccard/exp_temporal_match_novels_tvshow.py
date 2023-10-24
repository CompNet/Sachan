# Temporal matching between novels and the TV show
#
# Author: Arthur Amalvy
# 12/06
from typing import List, Literal
import os, sys, argparse, glob, pickle
import networkx as nx
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from plot_alignment_commons import graph_similarity_matrix, get_episode_i


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")


def get_align_matrix(
    M_sim: np.ndarray, M_block_to_episode: np.ndarray, threshold: float
) -> np.ndarray:
    """Given similarity between blocks and chapters, return the
    mapping between chapters and episodes.

    :param M_sim: ``(chapters_nb, blocks_nb)``
    :param M_block_to_episode: ``(blocks_nb)``
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


if __name__ == "__main__":

    FONTSIZE = 12
    CHAPTERS_NB = 344
    EPISODES_NB = 50

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gold-alignment-path", type=str)
    args = parser.parse_args()

    novel_graphs = []
    for path in sorted(glob.glob(f"{root_dir}/in/novels/instant/*.graphml")):
        novel_graphs.append(nx.read_graphml(path))
    assert len(novel_graphs) == CHAPTERS_NB
    novel_graphs = [
        nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
        for G in novel_graphs
    ]

    with open(args.gold_alignment_path, "rb") as f:
        M_align_gold = pickle.load(f)
        # the gold alignment matrix should contain alignment up to
        # season 6
        assert M_align_gold.shape == (60, CHAPTERS_NB)
    M_align_gold = M_align_gold[:EPISODES_NB, :]

    M_aligns = []
    M_block_to_episode_list = []
    M_sims = []
    best_thresholds = []

    block_methods: List[Literal["locations", "similarity"]] = [
        "locations",
        "similarity",
    ]

    thresholds = np.arange(0, 1, 0.01)

    for block_method in block_methods:
        tvshow_graphs = []
        for path in sorted(
            glob.glob(f"{root_dir}/in/tvshow/instant/block_{block_method}/*.graphml")
        ):
            tvshow_graphs.append(nx.read_graphml(path))

        # ignore season 7 and 8
        tvshow_graphs = [G for G in tvshow_graphs if G.graph["season"] < 7]

        # relabeling
        tvshow_graphs = [
            nx.relabel_nodes(
                G, {node: data["name"] for node, data in G.nodes(data=True)}
            )
            for G in tvshow_graphs
        ]

        M_sim = graph_similarity_matrix(
            novel_graphs, tvshow_graphs, "edges", use_weights=True
        )
        assert M_sim.shape == (len(novel_graphs), len(tvshow_graphs))
        M_sims.append(M_sim)

        M_block_to_episode = np.array([get_episode_i(G) for G in tvshow_graphs])
        M_block_to_episode_list.append(M_block_to_episode)

        best_f1 = 0.0
        best_M_align = None
        best_threshold = 0.0
        for threshold in thresholds:
            M_align = get_align_matrix(M_sim, M_block_to_episode, threshold)
            assert M_align.shape == (EPISODES_NB, CHAPTERS_NB)
            precision, recall, f1, _ = precision_recall_fscore_support(
                M_align_gold.flatten(),
                M_align.flatten(),
                average="binary",
                zero_division=0.0,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_M_align = M_align
                best_threshold = threshold

        M_aligns.append(best_M_align)
        best_thresholds.append(best_threshold)

    # Plotting
    # --------
    fig, axs = plt.subplots(1 + len(block_methods), 1, figsize=(16, 12))

    # plot alignment
    axs[0].set_title("Gold alignment", fontsize=FONTSIZE)
    axs[0].imshow(M_align_gold)
    axs[0].set_xlabel("chapters", fontsize=FONTSIZE)
    axs[0].set_ylabel("episodes", fontsize=FONTSIZE)
    for i, (block_method, M_align, threshold) in enumerate(
        zip(block_methods, M_aligns, best_thresholds)
    ):
        axs[i + 1].set_title(
            f"jaccard-index based alignment ({block_method}, threshold={threshold})",
            fontsize=FONTSIZE,
        )
        axs[i + 1].imshow(M_align)
        axs[i + 1].set_xlabel("chapters", fontsize=FONTSIZE)
        axs[i + 1].set_ylabel("episodes", fontsize=FONTSIZE)

    # Plot metrics figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    thresholds = np.arange(0, 1, 0.01)
    for (block_method, M_sim, M_block_to_episode, ax) in zip(
        block_methods, M_sims, M_block_to_episode_list, axs
    ):
        metrics = []
        for threshold in thresholds:
            M_align = get_align_matrix(M_sim, M_block_to_episode, threshold)
            precision, recall, f1, _ = precision_recall_fscore_support(
                M_align_gold.flatten(),
                M_align.flatten(),
                average="binary",
                zero_division=0.0,
            )
            metrics.append((precision, recall, f1))

        ax.plot(thresholds, [m[0] for m in metrics], label="precision")
        ax.plot(thresholds, [m[1] for m in metrics], label="recall")
        ax.plot(thresholds, [m[2] for m in metrics], label="f1")
        ax.set_title(block_method, fontsize=FONTSIZE)
        ax.legend(fontsize=FONTSIZE)

    # Show both figures
    plt.show()
