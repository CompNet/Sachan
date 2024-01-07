# Plot the predicted alignment between two medias
#
#
# Example usage:
#
# python plot_alignment.py
#
#
# For more details, see:
#
# python plot_alignment.py --help
#
#
# Author: Arthur Amalvy
import argparse
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    graph_similarity_matrix,
    load_medias_gold_alignment,
    load_medias_graphs,
    get_episode_i,
    load_medias_summaries,
    semantic_similarity,
    threshold_align_blocks,
    MEDIAS_STRUCTURAL_THRESHOLD,
    MEDIAS_SEMANTIC_THRESHOLD,
    MEDIAS_COMBINED_THRESHOLD,
    combined_similarities,
)
from smith_waterman import (
    smith_waterman_align_affine_gap,
    MEDIAS_SMITH_WATERMAN_STRUCTURAL_PARAMS,
    MEDIAS_SMITH_WATERMAN_SEMANTIC_PARAMS,
    MEDIAS_SMITH_WATERMAN_COMBINED_PARAMS,
)


if __name__ == "__main__":
    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    parser = argparse.ArgumentParser()
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
        help="one of 'structural', 'semantic' or 'combined'",
    )
    parser.add_argument(
        "-sf",
        "--similarity_function",
        type=str,
        default="tfidf",
        help="One of 'tfidf', 'sbert'.",
    )
    parser.add_argument(
        "-a",
        "--alignment",
        type=str,
        default="threshold",
        help="one of 'threshold', 'smith-waterman'",
    )
    parser.add_argument("-m1", "--min-delimiter-first-media", type=int, default=None)
    parser.add_argument("-x1", "--max-delimiter-first-media", type=int, default=None)
    parser.add_argument("-m2", "--min-delimiter-second-media", type=int, default=None)
    parser.add_argument("-x2", "--max-delimiter-second-media", type=int, default=None)
    parser.add_argument("-b", "--blocks", action="store_true")
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    G = load_medias_gold_alignment(
        args.medias,
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
    )

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
        )

        # Compute similarity
        # ------------------
        S = graph_similarity_matrix(
            first_media_graphs, second_media_graphs, "edges", True, "common+named"
        )

        if args.alignment == "threshold":
            if args.blocks:
                assert args.medias.startswith("tvshow")
                block_to_episode = np.array(
                    [get_episode_i(G) for G in first_media_graphs]
                )
                M = threshold_align_blocks(
                    S, MEDIAS_STRUCTURAL_THRESHOLD[args.medias], block_to_episode
                )
            else:
                M = S > MEDIAS_STRUCTURAL_THRESHOLD[args.medias]

        elif args.alignment == "smith-waterman":
            if args.blocks:
                raise NotImplementedError
            M, *_ = smith_waterman_align_affine_gap(
                S, **MEDIAS_SMITH_WATERMAN_STRUCTURAL_PARAMS[args.medias]
            )

        else:
            raise ValueError(f"unknown alignment method: {args.alignment}")

        f1 = precision_recall_fscore_support(
            G.flatten(), M.flatten(), average="binary", zero_division=0.0
        )[2]
        print(f"{f1=}")

        # Plot
        # ----
        plt.style.use("science")
        fig, ax = plt.subplots()
        fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)
        ax.set_title(f"F1 = {f1:.2f}", fontsize=FONTSIZE)
        ax.imshow(M, interpolation="none")
        first_media, second_media = args.medias.split("-")
        ax.set_xlabel(second_media, fontsize=FONTSIZE)
        ax.set_ylabel(first_media, fontsize=FONTSIZE)

        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    elif args.similarity == "semantic":
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
        S = semantic_similarity(
            first_summaries, second_summaries, args.similarity_function
        )

        if args.alignment == "threshold":
            M = S > MEDIAS_SEMANTIC_THRESHOLD[args.medias][args.similarity_function]
        elif args.alignment == "smith-waterman":
            M, *_ = smith_waterman_align_affine_gap(
                S,
                **MEDIAS_SMITH_WATERMAN_SEMANTIC_PARAMS[args.medias][
                    args.similarity_function
                ],
            )
        else:
            raise ValueError(f"unknown alignment method: {args.alignment}")

        f1 = precision_recall_fscore_support(
            G.flatten(), M.flatten(), average="binary", zero_division=0.0
        )[2]
        print(f"{f1=}")

        # Plot
        # ----
        plt.style.use("science")
        fig, ax = plt.subplots()
        fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)
        ax.set_title(f"F1 = {f1:.2f}", fontsize=FONTSIZE)
        ax.imshow(M, interpolation="none")
        first_media, second_media = args.medias.split("-")
        ax.set_xlabel(second_media, fontsize=FONTSIZE)
        ax.set_ylabel(first_media, fontsize=FONTSIZE)

        plt.tight_layout()
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
        S_structural = graph_similarity_matrix(
            tvshow_graphs, novels_graphs, "edges", True
        )
        S_semantic = semantic_similarity(
            first_summaries, second_summaries, args.similarity_function
        )
        S_combined = combined_similarities(S_structural, S_semantic)

        # Combination
        # -----------
        if args.alignment == "threshold":
            threshold = MEDIAS_COMBINED_THRESHOLD[args.medias][args.similarity_function]
            M = S_combined > threshold
        elif args.alignment == "smith-waterman":
            M, *_ = smith_waterman_align_affine_gap(
                S_semantic + S_structural,
                **MEDIAS_SMITH_WATERMAN_COMBINED_PARAMS[args.medias][
                    args.similarity_function
                ],
            )
        else:
            raise ValueError(f"unknown alignment method: {args.alignment}")

        f1 = precision_recall_fscore_support(
            G.flatten(), M.flatten(), average="binary", zero_division=0.0
        )[2]
        print(f"{f1=}")

        # Plot
        # ----
        plt.style.use("science")
        fig, ax = plt.subplots()
        fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)
        ax.set_title(f"F1 = {f1:.2f}", fontsize=FONTSIZE)
        ax.imshow(M, interpolation="none")
        ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
        ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)

        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    else:
        raise ValueError(f"unknown alignment method: {args.alignment}")
