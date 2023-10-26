# Plot the performance of a specific configuration of semantic
# alignment through time
#
#
# Example usage:
#
# python plot_semantic_alignment_through_time.py\
# --gold-alignment ./tvshow_novels_gold_alignment.pickle\
# --episode-summaries ./tvshow_episodes_summaries.txt\
# --chapter-summaries ./chapter_summaries.txt
#
#
# For more details, see:
#
# python plot_structural_alignment_through_time.py --help
#
#
# Author: Arthur Amalvy
from typing import List, Literal
import argparse, pickle
import scienceplots
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from plot_alignment_commons import (
    find_best_alignment,
    NOVEL_LIMITS,
    TVSHOW_SEASON_LIMITS,
    semantic_similarity,
)


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gold-alignment",
        type=str,
        default=None,
        help="Path to the gold chapters/episodes alignment. Must be specified if --best-treshold is specified.",
    )
    parser.add_argument(
        "-e",
        "--episode-summaries",
        type=str,
        help="Path to a file with episode summaries",
    )
    parser.add_argument(
        "-c",
        "--chapter-summaries",
        type=str,
        help="Path to a file with chapter summaries",
    )
    parser.add_argument("-o", "--output", type=str, help="Output file.")
    args = parser.parse_args()

    with open(args.chapter_summaries) as f:
        chapter_summaries = f.read().split("\n\n")[:-1]
    assert len(chapter_summaries) == NOVEL_LIMITS[-1]

    with open(args.episode_summaries) as f:
        episode_summaries = f.read().split("\n\n")
    episode_summaries = episode_summaries[: TVSHOW_SEASON_LIMITS[5]]
    assert len(episode_summaries) == TVSHOW_SEASON_LIMITS[5]

    with open(args.gold_alignment, "rb") as f:
        G = pickle.load(f)
    G = G[: TVSHOW_SEASON_LIMITS[-1], : NOVEL_LIMITS[-1]]
    assert G.shape == (TVSHOW_SEASON_LIMITS[5], NOVEL_LIMITS[-1])

    plt.style.use("science")
    fig, ax = plt.subplots()
    fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3)
    ax.set_xlabel("Seasons", fontsize=FONTSIZE)
    ax.set_ylabel("F1-score", fontsize=FONTSIZE)
    ax.grid()

    for similarity_function in ("tfidf", "sbert"):

        S = semantic_similarity(
            episode_summaries, chapter_summaries, similarity_function  # type: ignore
        )
        _, _, M = find_best_alignment(G, S)

        season_f1s = []

        for season in range(1, 7):

            limits = [0] + TVSHOW_SEASON_LIMITS
            start = limits[season - 1]
            end = limits[season]
            G_season = G[start:end, :]
            M_season = M[start:end, :]

            _, _, f1, _ = precision_recall_fscore_support(
                G_season.flatten(),
                M_season.flatten(),
                average="binary",
                zero_division=0.0,
            )
            season_f1s.append(f1)

        ax.plot(list(range(1, 7)), season_f1s, label=similarity_function)

    ax.legend()
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
