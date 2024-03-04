# Compute and plot alignment performance through time
#
# For more details, see:
#
# python plot_alignment_perf_through_time.py --help
#
#
# Author: Arthur Amalvy
import argparse, os, sys, json
from typing import Dict, Literal, Union
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots
from alignment_commons import (
    load_medias_gold_alignment,
    textual_similarity,
    graph_similarity_matrix,
    load_medias_graphs,
    load_novels_chapter_summaries,
    load_tvshow_episode_summaries,
    get_episode_i,
    TVSHOW_SEASON_LIMITS,
    threshold_align_blocks,
    tune_alpha_other_medias,
    tune_threshold_other_medias,
    combined_similarities,
    get_comics_chapter_issue_i,
)
from smith_waterman import (
    smith_waterman_align_affine_gap,
    tune_smith_waterman_params_other_medias,
    smith_waterman_align_blocks,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")


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
    parser.add_argument("-o", "--output", type=str, help="Output file.")
    args = parser.parse_args()

    structural_kwargs = json.loads(args.structural_kwargs)

    G = load_medias_gold_alignment(
        "tvshow-novels",
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
    )

    if args.similarity == "structural":
        first_media_graphs, second_media_graphs = load_medias_graphs(
            args.medias,
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
            "locations" if args.blocks else None,
            cumulative=args.cumulative,
        )

        S = graph_similarity_matrix(
            first_media_graphs,
            second_media_graphs,
            structural_kwargs["mode"],
            structural_kwargs["use_weights"],
            structural_kwargs["character_filtering"],
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
                if args.medias == "tvshow-novels":
                    block_to_narrunit = np.array(
                        [get_episode_i(G) for G in first_media_graphs]
                    )
                else:
                    assert args.medias == "comics-novels"
                    block_to_narrunit = np.array(
                        [get_comics_chapter_issue_i(G) for G in first_media_graphs]
                    )
                M = smith_waterman_align_blocks(
                    S,
                    block_to_narrunit,
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
                G_season.flatten(),
                M_season.flatten(),
                average="binary",
                zero_division=0.0,
            )
            season_f1s.append(f1)

        plt.style.use("science")
        fig, ax = plt.subplots()
        fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3)
        ax.plot(
            list(
                range(
                    args.min_delimiter_first_media, args.max_delimiter_first_media + 1
                )
            ),
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

    elif args.similarity == "textual":
        assert args.medias == "tvshow-novels"

        episode_summaries = load_tvshow_episode_summaries(
            args.min_delimiter_first_media, args.max_delimiter_first_media
        )
        chapter_summaries = load_novels_chapter_summaries(
            args.min_delimiter_second_media, args.max_delimiter_second_media
        )

        plt.style.use("science")
        fig, ax = plt.subplots()
        fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3)
        ax.set_xlabel("Seasons", fontsize=FONTSIZE)
        ax.set_ylabel("F1-score", fontsize=FONTSIZE)
        ax.grid()

        S = textual_similarity(
            episode_summaries, chapter_summaries, similarity_function  # type: ignore
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
                G_season.flatten(),
                M_season.flatten(),
                average="binary",
                zero_division=0.0,
            )
            season_f1s.append(f1)

        ax.plot(
            list(
                range(
                    args.min_delimiter_first_media,
                    args.max_delimiter_first_media + 1,
                )
            ),
            season_f1s,
        )

        ax.legend()
        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    elif args.similarity == "combined":
        assert not args.blocks

        tvshow_graphs, novels_graphs = load_medias_graphs(
            "tvshow-novels",
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
            "locations" if args.blocks else None,
            cumulative=args.cumulative,
        )

        episode_summaries = load_tvshow_episode_summaries(
            args.min_delimiter_first_media, args.max_delimiter_first_media
        )
        chapter_summaries = load_novels_chapter_summaries(
            args.min_delimiter_second_media, args.max_delimiter_second_media
        )

        plt.style.use("science")
        fig, ax = plt.subplots()
        fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.3)
        ax.set_xlabel("Seasons", fontsize=FONTSIZE)
        ax.set_ylabel("F1-score", fontsize=FONTSIZE)
        ax.grid()

        # Compute both similarities
        # -------------------------
        S_textual = textual_similarity(
            episode_summaries, chapter_summaries, args.similarity_function
        )

        S_structural = graph_similarity_matrix(
            tvshow_graphs, novels_graphs, "edges", True, "common"
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
                G_season.flatten(),
                M_season.flatten(),
                average="binary",
                zero_division=0.0,
            )
            season_f1s.append(f1)

        ax.plot(
            list(
                range(
                    args.min_delimiter_first_media,
                    args.max_delimiter_first_media + 1,
                )
            ),
            season_f1s,
        )

        ax.legend()
        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    else:
        raise ValueError(f"unknown alignment method: {args.similarity}")
