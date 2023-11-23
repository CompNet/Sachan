# Plot the performance of a specific configuration of semantic
# alignment through time
#
#
# Example usage:
#
# python plot_semantic_alignment_perf_through_time.py
#
#
# For more details, see:
#
# python plot_semantic_alignment_perf_through_time.py --help
#
#
# Author: Arthur Amalvy
import argparse
import scienceplots
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    find_best_alignment,
    semantic_similarity,
    load_medias_gold_alignment,
    load_novels_chapter_summaries,
    load_tvshow_episode_summaries,
    TVSHOW_SEASON_LIMITS,
)


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--min-delimiter-first-media", type=int, default=1)
    parser.add_argument("-x1", "--max-delimiter-first-media", type=int, default=6)
    parser.add_argument("-m2", "--min-delimiter-second-media", type=int, default=1)
    parser.add_argument("-x2", "--max-delimiter-second-media", type=int, default=5)
    parser.add_argument("-o", "--output", type=str, help="Output file.")
    args = parser.parse_args()

    episode_summaries = load_tvshow_episode_summaries(
        args.min_delimiter_first_media, args.max_delimiter_first_media
    )
    chapter_summaries = load_novels_chapter_summaries(
        args.min_delimiter_second_media, args.max_delimiter_second_media
    )

    G = load_medias_gold_alignment(
        "tvshow-novels",
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
    )

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

        for season in range(
            args.min_delimiter_first_media, args.max_delimiter_first_media + 1
        ):

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

        ax.plot(
            list(
                range(
                    args.min_delimiter_first_media, args.max_delimiter_first_media + 1
                )
            ),
            season_f1s,
            label=similarity_function,
        )

    ax.legend()
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
