# Plot the combined (semantic + structural) alignment between the
# novels and the TV show.
#
#
# Example usage:
#
# python plot_combined_alignment.py
#
#
# For more details, see:
#
# python plot_combined_alignment.py --help
#
#
# Author: Arthur Amalvy
import argparse, os, sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import scienceplots
import matplotlib.pyplot as plt
from tqdm import tqdm
from alignment_commons import (
    load_medias_graphs,
    semantic_similarity,
    graph_similarity_matrix,
    load_medias_gold_alignment,
    load_novels_chapter_summaries,
    load_tvshow_episode_summaries,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    CHAPTERS_NB = 344
    EPISODES_NB = 60

    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--min-delimiter-first-media", type=int, default=1)
    parser.add_argument("-x1", "--max-delimiter-first-media", type=int, default=6)
    parser.add_argument("-m2", "--min-delimiter-second-media", type=int, default=None)
    parser.add_argument("-x2", "--max-delimiter-second-media", type=int, default=None)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    # Load summaries
    # --------------
    episode_summaries = load_tvshow_episode_summaries(
        args.min_delimiter_first_media, args.max_delimiter_first_media
    )
    chapter_summaries = load_novels_chapter_summaries(
        args.min_delimiter_second_media, args.max_delimiter_second_media
    )

    # Load networks
    # -------------
    tvshow_graphs, novels_graphs = load_medias_graphs(
        "tvshow-novels",
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
    )

    # Compute similarity
    # ------------------
    S_semantic = semantic_similarity(episode_summaries, chapter_summaries, "sbert")
    S_structural = graph_similarity_matrix(tvshow_graphs, novels_graphs, "edges", True)

    # Load gold alignment
    # -------------------
    G = load_medias_gold_alignment(
        args.medias,
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
    )

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
    plt.style.use("science")
    fig, ax = plt.subplots()
    fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)
    ax.set_title(f"α = {best_alpha}, t = {best_t}, F1 = {best_f1}", fontsize=FONTSIZE)
    ax.imshow(best_M_align, interpolation="none")
    ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
    ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
