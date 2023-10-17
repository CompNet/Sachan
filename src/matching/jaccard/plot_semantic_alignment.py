# Plot the semantic alignment between the novels and the TV show
#
#
# Example usage:
#
# python plot_semantic_alignment.py\
# --chapter-summaries './chapter_summaries.txt'\
# --episode-summaries './episodes_summaries.txt'\
# --similarity-function sbert
#
#
# For more details, see:
#
# python plot_semantic_alignment.py --help
#
#
# Author: Arthur Amalvy
import argparse, pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from temporal_match_commons import semantic_similarity


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 3.0315

    CHAPTERS_NB = 344
    EPISODES_NB = 50

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
        "-s", "--similarity-function", type=str, help="Either 'tfidf' or 'sbert'."
    )
    parser.add_argument(
        "-t",
        "--best-threshold",
        action="store_true",
        help="If specified, plot the similarity matrix with the best threshold given the gold matchin. If specified, --gold-alignment must be specified as well.",
    )
    parser.add_argument(
        "-g",
        "--gold-alignment",
        type=str,
        default=None,
        help="Path to the gold chapters/episodes alignment. Must be specified if --best-treshold is specified.",
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

    # Compute similarity
    # ------------------
    S = semantic_similarity(
        episode_summaries, chapter_summaries, args.similarity_function
    )

    # Compute best threshold if necessary
    if args.best_threshold:

        with open(args.gold_alignment, "rb") as f:
            G = pickle.load(f)
        G = G[:EPISODES_NB, :CHAPTERS_NB]
        assert G.shape == (EPISODES_NB, CHAPTERS_NB)

        best_t = 0.0
        best_f1 = 0.0
        best_S_align = S > 0.0
        for t in np.arange(0.0, 1.0, 0.01):
            S_align = S > t
            _, _, f1, _ = precision_recall_fscore_support(
                G.flatten(), S_align.flatten(), average="binary", zero_division=0.0
            )
            if f1 > best_f1:
                best_t = t
                best_f1 = f1
                best_S_align = S_align
        print(f"{best_f1=}")
        print(f"{best_t=}")

    # Plot
    # ----
    fig, ax = plt.subplots()
    fig.set_size_inches(COLUMN_WIDTH_IN * 2, COLUMN_WIDTH_IN * 2 * 0.6)
    if args.best_threshold:
        ax.set_title(f"t = {best_t}, F1 = {best_f1}", fontsize=FONTSIZE)
        ax.imshow(best_S_align, interpolation="none")
    else:
        ax.imshow(S, interpolation="none")
    ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
    ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
