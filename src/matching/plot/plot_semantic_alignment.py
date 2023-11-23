# Plot the semantic alignment between the novels and the TV show
#
#
# Example usage:
#
# python plot_semantic_alignment.py --similarity-function sbert
#
#
# For more details, see:
#
# python plot_semantic_alignment.py --help
#
#
# Author: Arthur Amalvy
import argparse
import matplotlib.pyplot as plt
from alignment_commons import (
    find_best_alignment,
    semantic_similarity,
    load_medias_gold_alignment,
    load_novels_chapter_summaries,
    load_tvshow_episode_summaries,
)


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--similarity-function", type=str, help="Either 'tfidf' or 'sbert'."
    )
    parser.add_argument(
        "-t",
        "--best-threshold",
        action="store_true",
        help="If specified, plot the similarity matrix with the best threshold given the gold matchin. If specified, --gold-alignment must be specified as well.",
    )
    parser.add_argument("-m1", "--min-delimiter-first-media", type=int, default=None)
    parser.add_argument("-x1", "--max-delimiter-first-media", type=int, default=None)
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

    # Compute similarity
    # ------------------
    S = semantic_similarity(
        episode_summaries, chapter_summaries, args.similarity_function
    )

    # Compute best threshold if necessary
    if args.best_threshold:

        G = load_medias_gold_alignment(
            "tvshow-novels",
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
        )

        best_t, best_f1, best_S_align = find_best_alignment(G, S)

        print(f"{best_f1=}")
        print(f"{best_t=}")

    # Plot
    # ----
    fig, ax = plt.subplots()
    fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)
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
