# Plot the performance of a specific configuration of structural
# alignment through time (=seasons). Works for either tvshow-novels or
# tvshow-comics.
#
#
# Example usage:
#
# python plot_structural_alignment_perf_through_time.py -m 'tvshow-novels'
#
#
# For more details, see:
#
# python plot_structural_alignment_perf_through_time.py --help
#
#
# Author: Arthur Amalvy
import argparse, pickle
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    find_best_alignment,
    find_best_blocks_alignment,
    graph_similarity_matrix,
    load_medias_gold_alignment,
    load_medias_graphs,
    get_episode_i,
    TVSHOW_SEASON_LIMITS,
)


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--medias",
        type=str,
        help="Medias on which to compute alignment. Either 'tvshow-comics' or 'tvshow-novels'",
    )
    parser.add_argument(
        "-j",
        "--jaccard-index",
        type=str,
        default="edges",
        help="How to compute Jaccard index. Either on 'nodes' or on 'edges'.",
    )
    parser.add_argument(
        "-u",
        "--unweighted",
        action="store_true",
        help="If specified, wont use weighted Jaccard index.",
    )
    parser.add_argument(
        "-c",
        "--character-filtering",
        type=str,
        default="common",
        help="How to filter character. One of 'none', 'common', 'named' or 'common+named'.",
    )
    parser.add_argument("-m1", "--min-delimiter-first-media", type=int, default=1)
    parser.add_argument("-x1", "--max-delimiter-first-media", type=int, default=6)
    parser.add_argument("-m2", "--min-delimiter-second-media", type=int, default=None)
    parser.add_argument("-x2", "--max-delimiter-second-media", type=int, default=None)
    parser.add_argument("-b", "--blocks", action="store_true")
    parser.add_argument("-o", "--output", type=str, help="Output file.")
    args = parser.parse_args()
    assert args.medias in ("tvshow-comics", "tvshow-novels")

    G = load_medias_gold_alignment(
        args.medias,
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
    )

    first_media_graphs, second_media_graphs = load_medias_graphs(
        args.medias,
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
    )

    S = graph_similarity_matrix(
        first_media_graphs,
        second_media_graphs,
        args.jaccard_index,
        not args.unweighted,
        args.character_filtering,
    )

    if args.blocks:
        assert args.medias.startswith("tvshow")
        block_to_episode = np.array([get_episode_i(G) for G in first_media_graphs])
        _, f1, M = find_best_blocks_alignment(G, S, block_to_episode)
    else:
        _, f1, M = find_best_alignment(G, S)

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
            G_season.flatten(), M_season.flatten(), average="binary", zero_division=0.0
        )
        season_f1s.append(f1)

    plt.style.use("science")
    fig, ax = plt.subplots()
    fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3)
    ax.plot(
        list(range(args.min_delimiter_first_media, args.max_delimiter_first_media + 1)),
        season_f1s,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Seasons", fontsize=FONTSIZE)
    ax.set_ylabel("F1-score", fontsize=FONTSIZE)
    ax.grid()

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
