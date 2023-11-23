# Compute the performance of the combined (semantic + structural)
# alignment between the novels and the TV show.
#
#
# Example usage:
#
# python compute_combined_alignment_performance.py --semantic-similarity-function sbert
#
#
# For more details, see:
#
# python compute_combined_alignment_performance.py --help
#
#
# Author: Arthur Amalvy
import argparse, os, sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    load_medias_gold_alignment,
    semantic_similarity,
    graph_similarity_matrix,
    load_medias_graphs,
    load_novels_chapter_summaries,
    load_tvshow_episode_summaries,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--semantic-similarity-function",
        type=str,
        default="tfidf",
        help="one of: 'tfidf', 'sbert'",
    )
    parser.add_argument("-m1", "--min-delimiter-first-media", type=int, default=1)
    parser.add_argument("-x1", "--max-delimiter-first-media", type=int, default=6)
    parser.add_argument("-m2", "--min-delimiter-second-media", type=int, default=1)
    parser.add_argument("-x2", "--max-delimiter-second-media", type=int, default=5)
    parser.add_argument("-b", "--blocks", action="store_true")
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

    episode_summaries = load_tvshow_episode_summaries(
        args.min_delimiter_first_media, args.max_delimiter_first_media
    )
    chapter_summaries = load_novels_chapter_summaries(
        args.min_delimiter_second_media, args.max_delimiter_second_media
    )

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
