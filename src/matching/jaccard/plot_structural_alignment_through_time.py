# Plot the performance of a specific configuration of structural
# alignment through time
#
#
# Example usage:
#
# python plot_structural_alignment_through_time.py --gold-alignment ./tvshow_novels_gold_alignment.pickle
#
#
# For more details, see:
#
# python plot_structural_alignment_through_time.py --help
#
#
# Author: Arthur Amalvy
import argparse, pickle
import scienceplots
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from plot_alignment_commons import (
    find_best_alignment,
    graph_similarity_matrix,
    load_novels_graphs,
    load_tvshow_graphs,
    NOVEL_LIMITS,
    TVSHOW_SEASON_LIMITS,
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
    parser.add_argument("-o", "--output", type=str, help="Output file.")
    args = parser.parse_args()

    with open(args.gold_alignment, "rb") as f:
        G = pickle.load(f)
    G = G[: TVSHOW_SEASON_LIMITS[-1], : NOVEL_LIMITS[-1]]
    assert G.shape == (TVSHOW_SEASON_LIMITS[5], NOVEL_LIMITS[-1])

    novels_graphs = load_novels_graphs()
    tvshow_graphs = load_tvshow_graphs(max_season=6)

    S = graph_similarity_matrix(
        tvshow_graphs,
        novels_graphs,
        args.jaccard_index,
        not args.unweighted,
        args.character_filtering,
    )

    _, f1, M = find_best_alignment(G, S)

    season_f1s = []

    for season in range(1, 7):

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
    ax.plot(list(range(1, 7)), season_f1s)
    ax.set_xlabel("Seasons", fontsize=FONTSIZE)
    ax.set_ylabel("F1-score", fontsize=FONTSIZE)
    ax.grid()

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
