# Plot the combined (semantic + structural) alignment between the
# novels and the TV show.
#
#
# Example usage:
#
# python plot_combined_alignment.py\
# --chapter-summaries './chapter_summaries.txt'\
# --episode-summaries './episodes_summaries.txt'\
# --similarity-function sbert
#
#
# For more details, see:
#
# python plot_combined_alignment.py --help
#
#
# Author: Arthur Amalvy
import argparse, pickle, glob, os, sys
import networkx as nx
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
from temporal_match_commons import semantic_similarity, graph_similarity_matrix

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 3.0315

    CHAPTERS_NB = 344
    EPISODES_NB = 60

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--chapter-summaries",
        type=str,
        help="Path to a file with chapter summaries",
    )
    parser.add_argument(
        "-e",
        "--episode-summaries",
        type=str,
        help="Path to a file with episode summaries",
    )
    parser.add_argument(
        "-g",
        "--gold-alignment",
        type=str,
        default=None,
        help="Path to the gold chapters/episodes alignment.",
    )
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    # Load summaries
    # --------------
    with open(args.chapter_summaries) as f:
        chapter_summaries = f.read().split("\n\n")[:-1]
    assert len(chapter_summaries) == CHAPTERS_NB

    with open(args.episode_summaries) as f:
        episode_summaries = f.read().split("\n\n")
    episode_summaries = episode_summaries[:EPISODES_NB]
    assert len(episode_summaries) == EPISODES_NB

    # Load networks
    # -------------
    chapter_graphs = []
    for path in sorted(glob.glob(f"{root_dir}/in/novels/instant/*.graphml")):
        chapter_graphs.append(nx.read_graphml(path))
    chapter_graphs = chapter_graphs[:CHAPTERS_NB]
    assert len(chapter_graphs) == CHAPTERS_NB

    episode_graphs = []
    for path in sorted(
        glob.glob(f"{root_dir}/in/tvshow/instant/block_locations/*.graphml")
    ):
        episode_graphs.append(nx.read_graphml(path))
    episode_graphs = episode_graphs[:EPISODES_NB]
    assert len(episode_graphs) == EPISODES_NB

    # Compute similarity
    # ------------------
    S_semantic = semantic_similarity(episode_summaries, chapter_summaries, "sbert")
    S_structural = graph_similarity_matrix(
        episode_graphs, chapter_graphs, "edges", True
    )

    with open(args.gold_alignment, "rb") as f:
        G = pickle.load(f)
    G = G[:EPISODES_NB, :CHAPTERS_NB]
    assert G.shape == (EPISODES_NB, CHAPTERS_NB)

    # Combination
    # -----------
    # Compute the best combination of both matrices
    # S_combined = α × S_semantic + (1 - α) × S_structural
    alphas = np.arange(0.0, 1.0, 0.01)
    ts = np.arange(0.0, 1.0, 0.01)
    f1s = np.zeros((alphas.shape[0], ts.shape[0]))

    for alpha_i, alpha in tqdm(enumerate(alphas), total=alphas.shape[0]):
        M_sim = alpha * S_semantic + (1 - alpha) * S_structural
        for t_i, t in enumerate(ts):
            M_align = M_sim > t
            precision, recall, f1, _ = precision_recall_fscore_support(
                G.flatten(),
                M_align.flatten(),
                average="binary",
                zero_division=0.0,
            )
            f1s[alpha_i][t_i] = f1

    best_f1_loc = np.argwhere(f1s == np.max(f1s))[0]
    best_f1 = np.max(f1s)
    best_alpha = best_f1_loc[0] / 100.0
    best_t = best_f1_loc[1] / 100.0
    best_M_sim = best_alpha * S_semantic + (1 - best_alpha) * S_structural
    best_M_align = best_M_sim > best_t
    print(f"{best_alpha=}")
    print(f"{best_t=}")
    print(f"{best_f1=}")

    # Plot
    # ----
    fig, ax = plt.subplots()
    fig.set_size_inches(COLUMN_WIDTH_IN * 2, COLUMN_WIDTH_IN * 2 * 0.6)
    ax.set_title(f"α = {best_alpha}, t = {best_t}, F1 = {best_f1}", fontsize=FONTSIZE)
    ax.imshow(best_M_align, interpolation="none")
    ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
    ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
