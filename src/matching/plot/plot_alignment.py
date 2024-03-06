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
import argparse, json
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    get_comics_chapter_issue_i,
    graph_similarity_matrix,
    load_medias_gold_alignment,
    load_medias_graphs,
    get_episode_i,
    load_medias_summaries,
    textual_similarity,
    threshold_align_blocks,
    combined_similarities,
    tune_alpha_other_medias,
    tune_threshold_other_medias,
)
from smith_waterman import (
    smith_waterman_align_affine_gap,
    smith_waterman_align_blocks,
    tune_smith_waterman_params_other_medias,
)


def to_error_matrix(G: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    :param G: gold alignment of shape (n, m)
    :param M: predicted alignment of shape (n, m)
    :return: a colored error matrix of shape (n, m, 3)
    """

    TN = 0
    TP = 1
    FN = 2
    FP = 3

    ERROR_COLORS = np.array(
        [
            [68, 1, 84],
            [85, 198, 103],
            [253, 231, 37],
            [192, 58, 131],
        ]
    )

    E = np.zeros(G.shape, dtype="int")
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            truth = G[i][j]
            pred = M[i][j]
            if truth == 0:
                E[i][j] = TN if pred == truth else FP
            else:
                E[i][j] = TP if pred == truth else FN

    return ERROR_COLORS[E]


if __name__ == "__main__":
    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

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
        "-c",
        "--cumulative",
        action="store_true",
        help="if specified, perform alignment using the cumulative networks (if applicable)",
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
            comics_blocks=bool(args.blocks),
            cumulative=args.cumulative,
        )

        # Compute similarity
        # ------------------
        structural_kwargs = json.loads(args.structural_kwargs)
        S = graph_similarity_matrix(
            first_media_graphs, second_media_graphs, **structural_kwargs
        )

        if args.alignment == "threshold":
            t = tune_threshold_other_medias(
                args.medias,
                "structural",
                np.arange(0.0, 1.0, 0.01),
                structural_mode=structural_kwargs["mode"],
                structural_use_weights=structural_kwargs["use_weights"],
                structural_filtering=structural_kwargs["character_filtering"],
            )
            print(f"found {t=}")
            if args.blocks:
                M = threshold_align_blocks(
                    args.medias, first_media_graphs, second_media_graphs, S, t
                )
            else:
                M = S > t

        elif args.alignment == "smith-waterman":
            (
                gap_start_penalty,
                gap_cont_penalty,
                neg_th,
            ) = tune_smith_waterman_params_other_medias(
                args.medias,
                "structural",
                np.arange(0.0, 0.2, 0.01),
                np.arange(0.0, 0.2, 0.01),
                np.arange(0.0, 0.1, 0.1),
                structural_mode=structural_kwargs["mode"],
                structural_use_weights=structural_kwargs["use_weights"],
                structural_filtering=structural_kwargs["character_filtering"],
            )
            print(f"found {gap_start_penalty=} {gap_cont_penalty=} {neg_th=}")
            if args.blocks:
                M = smith_waterman_align_blocks(
                    args.medias,
                    first_media_graphs,
                    second_media_graphs,
                    S,
                    gap_start_penalty=gap_start_penalty,
                    gap_cont_penalty=gap_cont_penalty,
                    neg_th=neg_th,
                )
            else:
                M, *_ = smith_waterman_align_affine_gap(
                    S, gap_start_penalty, gap_cont_penalty, neg_th
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
        E = to_error_matrix(G, M)
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3))
        ax.set_title(f"F1 = {f1*100:.2f}", fontsize=FONTSIZE)
        ax.imshow(E, interpolation="none", aspect="auto")
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

        plt.tight_layout()
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

        if args.alignment == "threshold":
            t = tune_threshold_other_medias(
                args.medias,
                "textual",
                np.arange(0.0, 1.0, 0.01),
                textual_sim_fn=args.similarity_function,
                silent=True,
            )
            M = S > t
        elif args.alignment == "smith-waterman":
            (
                gap_start_penalty,
                gap_cont_penalty,
                neg_th,
            ) = tune_smith_waterman_params_other_medias(
                args.medias,
                "textual",
                np.arange(0.0, 0.2, 0.01),
                np.arange(0.0, 0.2, 0.01),
                np.arange(0.0, 0.1, 0.1),
                textual_sim_fn=args.similarity_function,
                silent=True,
            )
            M, *_ = smith_waterman_align_affine_gap(
                S,
                gap_start_penalty=gap_start_penalty,
                gap_cont_penalty=gap_cont_penalty,
                neg_th=neg_th,
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
        E = to_error_matrix(G, M)
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3))
        ax.set_title(f"F1 = {f1*100:.2f}", fontsize=FONTSIZE)
        ax.imshow(E, interpolation="none", aspect="auto")
        if args.medias == "comics-novels":
            ax.set_ylabel("Comics Issues", fontsize=FONTSIZE)
            ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
        elif args.medias == "tvshow-comics":
            ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
            ax.set_xlabel("Comics Issues", fontsize=FONTSIZE)
        elif args.medias == "tvshow-novels":
            ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
            ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)

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
            cumulative=args.cumulative,
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

        # Combination
        # -----------
        if args.alignment == "threshold":
            alpha, t = tune_alpha_other_medias(
                args.medias,
                "threshold",
                np.arange(0.0, 1.0, 0.01),  # alpha
                [np.arange(0.0, 1.0, 0.01)],  # threshold
                textual_sim_fn=args.similarity_function,
                structural_mode=structural_kwargs["mode"],
                structural_use_weights=structural_kwargs["use_weights"],
                structural_filtering=structural_kwargs["character_filtering"],
                silent=True,
            )
            S_combined = combined_similarities(S_structural, S_textual, alpha)
            M = S_combined > t
        elif args.alignment == "smith-waterman":
            (
                alpha,
                gap_start_penalty,
                gap_cont_penalty,
                neg_th,
            ) = tune_alpha_other_medias(
                args.medias,
                "smith-waterman",
                np.arange(0.1, 0.9, 0.05),  # alpha
                [
                    np.arange(0.0, 0.2, 0.01),  # gap_start_penalty
                    np.arange(0.0, 0.2, 0.01),  # gap_cont_penalty
                    np.arange(0.0, 0.1, 0.1),  # neg_th
                ],
                textual_sim_fn=args.similarity_function,
                structural_mode=structural_kwargs["mode"],
                structural_use_weights=structural_kwargs["use_weights"],
                structural_filtering=structural_kwargs["character_filtering"],
                silent=True,
            )
            S_combined = combined_similarities(S_structural, S_textual, alpha)
            M, *_ = smith_waterman_align_affine_gap(
                S_combined, gap_start_penalty, gap_cont_penalty, neg_th
            )
        else:
            raise ValueError(f"unknown alignment method: {args.alignment}")

        f1 = precision_recall_fscore_support(
            G.flatten(), M.flatten(), average="binary", zero_division=0.0
        )[2]
        print(f"{f1=}")

        # Plot
        # ----
        E = to_error_matrix(G, M)
        plt.style.use("science")
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3))
        ax.set_title(f"F1 = {f1*100:.2f}", fontsize=FONTSIZE)
        ax.imshow(E, interpolation="none", aspect="auto")
        if args.medias == "comics-novels":
            ax.set_ylabel("Comics Issues", fontsize=FONTSIZE)
            ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
        elif args.medias == "tvshow-comics":
            ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
            ax.set_xlabel("Comics Issues", fontsize=FONTSIZE)
        elif args.medias == "tvshow-novels":
            ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
            ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)

        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    else:
        raise ValueError(f"unknown alignment method: {args.alignment}")
