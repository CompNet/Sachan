# Plot the similarity matrix between two medias
#
#
# Example usage:
#
# python plot_similarity.py
#
#
# For more details, see:
#
# python plot_similarity.py --help
#
#
# Author: Arthur Amalvy
import argparse, json
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from alignment_commons import (
    graph_similarity_matrix,
    load_medias_graphs,
    load_medias_summaries,
    textual_similarity,
    combined_similarities,
)


if __name__ == "__main__":
    FONTSIZE = 8
    COLUMN_WIDTH_IN = 5.166
    H_W_RATIO = 0.2

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--medias",
        type=str,
        help="Medias on which to compute alignment. One of 'tvshow-comics', 'tvshow-novels', 'comics-novels'",
    )
    parser.add_argument(
        "-s",
        "--similarity",
        type=str,
        default="structural",
        help="one of 'structural', 'textual' or 'combined'",
    )
    parser.add_argument(
        "-sf",
        "--similarity-function",
        type=str,
        default="tfidf",
        help="One of 'tfidf', 'sbert'.",
    )
    parser.add_argument(
        "-sk",
        "--structural-kwargs",
        type=str,
        default='{"mode": "edges", "use_weights": true, "character_filtering": "named"}',
        help="JSON formatted kwargs for structural alignment",
    )
    parser.add_argument("-a", "--alpha", type=float, default=0.5)
    parser.add_argument("-m1", "--min-delimiter-first-media", type=int, default=None)
    parser.add_argument("-x1", "--max-delimiter-first-media", type=int, default=None)
    parser.add_argument("-m2", "--min-delimiter-second-media", type=int, default=None)
    parser.add_argument("-x2", "--max-delimiter-second-media", type=int, default=None)
    parser.add_argument("-b", "--blocks", action="store_true")
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    plt.style.use("science")

    if args.similarity == "structural":
        # Load graphs
        # -----------
        first_media_graphs, second_media_graphs = load_medias_graphs(
            args.medias,
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
            "locations" if args.blocks else None,
            comics_blocks=bool(args.blocks),
        )

        # Compute similarity
        # ------------------
        structural_kwargs = json.loads(args.structural_kwargs)
        S = graph_similarity_matrix(
            first_media_graphs, second_media_graphs, **structural_kwargs
        )

        # Plot
        # ----
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * H_W_RATIO))
        ax.imshow(S, interpolation="none", aspect="auto")
        first_media, second_media = args.medias.split("-")
        if args.medias == "comics-novels":
            ax.set_ylabel("Comics Issues", fontsize=FONTSIZE)
            ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
        elif args.medias == "tvshow-comics":
            ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
            ax.set_xlabel("Comics Issues", fontsize=FONTSIZE)
        elif args.medias == "tvshow-novels":
            ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
            ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)

        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    elif args.similarity == "textual":
        # Load summaries
        # --------------
        first_summaries, second_summaries = load_medias_summaries(
            args.medias,
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
        )

        # Compute similarity
        # ------------------
        S = textual_similarity(
            first_summaries, second_summaries, args.similarity_function
        )

        # Plot
        # ----
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * H_W_RATIO))
        ax.imshow(S, interpolation="none", aspect="auto")
        if args.medias == "comics-novels":
            ax.set_ylabel("Comics Issues", fontsize=FONTSIZE)
            ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
        elif args.medias == "tvshow-comics":
            ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
            ax.set_xlabel("Comics Issues", fontsize=FONTSIZE)
        elif args.medias == "tvshow-novels":
            ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
            ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)

        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    elif args.similarity == "combined":
        assert not args.blocks

        # Load summaries
        # --------------
        first_summaries, second_summaries = load_medias_summaries(
            args.medias,
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
        )

        # Load networks
        # -------------
        tvshow_graphs, novels_graphs = load_medias_graphs(
            args.medias,
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
        )

        # Compute similarity
        # ------------------
        structural_kwargs = json.loads(args.structural_kwargs)
        S_structural = graph_similarity_matrix(
            tvshow_graphs, novels_graphs, **structural_kwargs
        )
        S_textual = textual_similarity(
            first_summaries, second_summaries, args.similarity_function
        )
        S_combined = combined_similarities(S_structural, S_textual, args.alpha)

        # Plot
        # ----
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * H_W_RATIO))
        ax.imshow(S_combined, interpolation="none", aspect="auto")
        if args.medias == "comics-novels":
            ax.set_ylabel("Comics Issues", fontsize=FONTSIZE)
            ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
        elif args.medias == "tvshow-comics":
            ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
            ax.set_xlabel("Comics Issues", fontsize=FONTSIZE)
        elif args.medias == "tvshow-novels":
            ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
            ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)

        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    else:
        raise ValueError(f"unknown similarity: {args.similarity}")
