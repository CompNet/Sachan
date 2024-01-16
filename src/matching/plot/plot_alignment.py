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
    graph_similarity_matrix,
    load_medias_gold_alignment,
    load_medias_graphs,
    get_episode_i,
    load_medias_summaries,
    textual_similarity,
    threshold_align_blocks,
    combined_similarities,
    tune_threshold_other_medias,
)
from smith_waterman import (
    smith_waterman_align_affine_gap,
    tune_smith_waterman_params_other_medias,
)


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
        "-sf",
        "--similarity_function",
        type=str,
        default="tfidf",
        help="One of 'tfidf', 'sbert'.",
    )
    parser.add_argument(
        "-sk",
        "--structural_kwargs",
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
                assert args.medias.startswith("tvshow")
                block_to_episode = np.array(
                    [get_episode_i(G) for G in first_media_graphs]
                )
                M = threshold_align_blocks(S, t, block_to_episode)
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
                raise NotImplementedError
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
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3))
        ax.set_title(f"F1 = {f1:.2f}", fontsize=FONTSIZE)
        ax.imshow(M, interpolation="none", aspect="auto")
        first_media, second_media = args.medias.split("-")
        ax.set_xlabel(second_media, fontsize=FONTSIZE)
        ax.set_ylabel(first_media, fontsize=FONTSIZE)

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
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3))
        ax.set_title(f"F1 = {f1:.2f}", fontsize=FONTSIZE)
        ax.imshow(M, interpolation="none", aspect="auto")
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
        S_textual = textual_similarity(
            first_summaries, second_summaries, args.similarity_function
        )
        S_combined = combined_similarities(S_structural, S_textual)

        # Combination
        # -----------
        if args.alignment == "threshold":
            t = tune_threshold_other_medias(
                args.medias,
                "combined",
                np.arange(0.0, 1.0, 0.01),
                textual_sim_fn=args.similarity_function,
                structural_mode=args.structural_kwargs["mode"],
                structural_use_weights=args.structural_kwargs["use_weights"],
                structural_filtering=args.structural_kwargs["filtering"],
                silent=True,
            )
            M = S_combined > t
        elif args.alignment == "smith-waterman":
            (
                gap_start_penalty,
                gap_cont_penalty,
                neg_th,
            ) = tune_smith_waterman_params_other_medias(
                args.medias,
                "combined",
                np.arange(0.0, 0.2, 0.01),
                np.arange(0.0, 0.2, 0.01),
                np.arange(0.0, 0.1, 0.1),  # effectively no search
                textual_sim_fn=args.similarity_function,
                structural_mode=args.structural_kwargs["mode"],
                structural_use_weights=args.structural_kwargs["use_weights"],
                structural_filtering=args.structural_kwargs["filtering"],
                silent=True,
            )
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
        plt.style.use("science")
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3))
        ax.set_title(f"F1 = {f1:.2f}", fontsize=FONTSIZE)
        ax.imshow(M, interpolation="none", aspect="auto")
        ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
        ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)

        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    else:
        raise ValueError(f"unknown alignment method: {args.alignment}")
