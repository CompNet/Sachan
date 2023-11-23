# Compute the performance of the combined (semantic + structural)
# alignment between the novels and the TV show.
#
#
# Example usage:
#
# python plot_combined_alignment_perf_through_time.py --semantic-similarity-function sbert
#
#
# For more details, see:
#
# python plot_combined_alignment_perf_through_time.py --help
#
#
# Author: Arthur Amalvy
import argparse, os, sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from alignment_commons import (
    load_medias_gold_alignment,
    semantic_similarity,
    graph_similarity_matrix,
    load_novels_graphs,
    load_tvshow_graphs,
    load_medias_graphs,
    load_novels_chapter_summaries,
    load_tvshow_episode_summaries,
    TVSHOW_SEASON_LIMITS,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--semantic-similarity-function",
        type=str,
        default="tfidf",
        help="one of: 'tfidf', 'sbert'",
    )
    parser.add_argument("-m1", "--min-delimiter-first-media", type=int, default=None)
    parser.add_argument("-x1", "--max-delimiter-first-media", type=int, default=None)
    parser.add_argument("-m2", "--min-delimiter-second-media", type=int, default=None)
    parser.add_argument("-x2", "--max-delimiter-second-media", type=int, default=None)
    parser.add_argument("-b", "--blocks", action="store_true")
    parser.add_argument("-o", "--output", type=str, help="Output file.")
    args = parser.parse_args()

    G = load_medias_gold_alignment(
        "tvshow-novels",
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
    )

    tvshow_graphs, novels_graphs = load_medias_graphs(
        "tvshow-novels",
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
    )

    novels_graphs = load_novels_graphs(args.min_novel, args.max_novel)

    tvshow_graphs = load_tvshow_graphs(
        args.min_season, args.max_season, blocks="locations" if args.blocks else None
    )

    episode_summaries = load_tvshow_episode_summaries(
        args.min_delimiter_first_media, args.max_delimiter_first_media
    )
    chapter_summaries = load_novels_chapter_summaries(
        args.min_delimiter_second_media, args.max_delimiter_second_media
    )

    # Compute both similarities
    # -------------------------
    S_semantic = semantic_similarity(
        episode_summaries, chapter_summaries, args.semantic_similarity_function
    )

    S_structural = graph_similarity_matrix(
        tvshow_graphs, novels_graphs, "edges", True, "common"
    )

    # Combination
    # -----------
    # Compute the best combination of both matrices
    # S_combined = α × S_semantic + (1 - α) × S_structural
    alphas = np.arange(0.0, 1.0, 0.01)
    ts = np.arange(0.0, 1.0, 0.01)
    f1s = np.zeros((alphas.shape[0], ts.shape[0]))

    print("searching for α and t...", file=sys.stderr)
    for alpha_i, alpha in tqdm(enumerate(alphas), total=alphas.shape[0]):
        S = alpha * S_semantic + (1 - alpha) * S_structural
        for t_i, t in enumerate(ts):
            M = S > t
            precision, recall, f1, _ = precision_recall_fscore_support(
                G.flatten(),
                M.flatten(),
                average="binary",
                zero_division=0.0,
            )
            f1s[alpha_i][t_i] = f1

    best_f1_loc = np.argwhere(f1s == np.max(f1s))[0]
    best_f1 = np.max(f1s)
    best_alpha = best_f1_loc[0] / 100.0
    best_t = best_f1_loc[1] / 100.0
    best_S = best_alpha * S_semantic + (1 - best_alpha) * S_structural
    best_M = best_S > best_t
    print(f"{best_alpha=}")
    print(f"{best_t=}")
    print(f"{best_f1=}")

    season_f1s = []

    for season in range(args.min_first_media_delimiter, args.max_first_media_delimiter):

        limits = [0] + TVSHOW_SEASON_LIMITS
        start = limits[season - 1]
        end = limits[season]
        G_season = G[start:end, :]
        M_season = best_M[start:end, :]

        _, _, f1, _ = precision_recall_fscore_support(
            G_season.flatten(), M_season.flatten(), average="binary", zero_division=0.0
        )
        season_f1s.append(f1)

    plt.style.use("science")
    fig, ax = plt.subplots()
    fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3)
    ax.plot(
        list(range(args.min_first_media_delimiter, args.max_first_media_delimiter)),
        season_f1s,
    )
    ax.set_xlabel("Seasons", fontsize=FONTSIZE)
    ax.set_ylabel("F1-score", fontsize=FONTSIZE)
    ax.grid()

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
