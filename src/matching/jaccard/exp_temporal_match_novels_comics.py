# Temporal matching between the novels and the comics
#
# Example usage: python exp_temporal_match_novels_comics.py -g ./novels_comics_gold_alignment.pickle
#
# Author: Arthur Amalvy
from typing import List, Tuple
import os, sys, glob, argparse, pickle
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.metrics import precision_recall_fscore_support
from temporal_match import graph_similarity_matrix


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")


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
    comics_graphs = [
        nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
        for G in comics_graphs
    ]
    assert len(comics_graphs) > 0

    # Compute gold alignment
    with open(args.gold_alignment_path, "rb") as f:
        # (novels_chapters_nb, comics_chapters_nb)
        M_align_gold = pickle.load(f)

    # (novels_chapters_nb, comics_chapters_nb)
    M_sim = graph_similarity_matrix(
        novels_graphs, comics_graphs, "edges", use_weights=True
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
    axs[0].set_xlabel("comics chapters", fontsize=FONTSIZE)
    axs[0].set_ylabel("novels chapters", fontsize=FONTSIZE)
    axs[1].imshow(best_M_align)
    axs[1].set_title(
        f"Jaccard similarity alignment (threshold: {best_threshold})", fontsize=FONTSIZE
    )
    axs[1].set_xlabel("comics chapters", fontsize=FONTSIZE)
    axs[1].set_ylabel("novels chapters", fontsize=FONTSIZE)

    # Plot precision, recall and F1 in a second figure
    plt.figure(2, figsize=(16, 10))
    plt.plot(thresholds, [m[0] for m in metrics], label="precision")
    plt.plot(thresholds, [m[1] for m in metrics], label="recall")
    plt.plot(thresholds, [m[2] for m in metrics], label="f1")
    plt.xlabel("Chapters similarity threshold", fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)

    # Show both figures
    plt.show()
