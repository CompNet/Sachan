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
from matplotlib.ticker import MaxNLocator
import scienceplots
from alignment_commons import (
    load_medias_gold_alignment,
    semantic_similarity,
    graph_similarity_matrix,
    load_medias_graphs,
    load_novels_chapter_summaries,
    load_tvshow_episode_summaries,
    find_best_combined_alignment,
    find_best_blocks_alignment,
    get_episode_i,
    find_best_alignment,
    TVSHOW_SEASON_LIMITS,
)
from smith_waterman import smith_waterman_align_affine_gap

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
        help="one of 'structural', 'semantic' or 'combined'",
    )
    parser.add_argument(
        "-a",
        "--alignment",
        type=str,
        default="threshold",
        help="one of 'threshold', 'smith-waterman'",
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

    if args.similarity == "structural":
        assert args.medias in ("tvshow-comics", "tvshow-novels")

        first_media_graphs, second_media_graphs = load_medias_graphs(
            args.medias,
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
            "locations" if args.blocks else None,
        )

        S = graph_similarity_matrix(
            first_media_graphs,
            second_media_graphs,
            args.jaccard_index,
            not args.unweighted,
            args.character_filtering,
        )

        if args.alignment == "threshold":
            if args.blocks:
                assert args.medias.startswith("tvshow")
                block_to_episode = np.array(
                    [get_episode_i(G) for G in first_media_graphs]
                )
                _, f1, M = find_best_blocks_alignment(G, S, block_to_episode)
            else:
                _, f1, M = find_best_alignment(G, S)
        elif args.alignment == "smith-waterman":
            if args.blocks:
                raise RuntimeError("unimplemented")
            # TODO: penalties are hardcoded as a test.
            M, *_ = smith_waterman_align_affine_gap(S, -0.5, -0.01, 0.1)
            f1 = precision_recall_fscore_support(
                G.flatten(), M.flatten(), average="binary", zero_division=0.0
            )[2]
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

    elif args.similarity == "semantic":
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

        for similarity_function in ("tfidf", "sbert"):
            S = semantic_similarity(
                episode_summaries, chapter_summaries, similarity_function  # type: ignore
            )

            if args.alignment == "threshold":
                _, _, M = find_best_alignment(G, S)
            elif args.alignment == "smith-waterman":
                # TODO: penalty are hardcoded as a test.
                M, *_ = smith_waterman_align_affine_gap(S, -0.5, -0.01, 0.1)
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
                label=similarity_function,
            )

        ax.legend()
        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    elif args.similarity == "combined":
        assert args.medias == "tvshow-novels"
        assert not args.blocks

        tvshow_graphs, novels_graphs = load_medias_graphs(
            "tvshow-novels",
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
            "locations" if args.blocks else None,
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

        for similarity_function in ("tfidf", "sbert"):
            # Compute both similarities
            # -------------------------
            S_semantic = semantic_similarity(
                episode_summaries, chapter_summaries, similarity_function
            )

            S_structural = graph_similarity_matrix(
                tvshow_graphs, novels_graphs, "edges", True, "common"
            )

            # Combination
            # -----------
            if args.alignment == "threshold":
                # Compute the best combination of both matrices
                # S_combined = α × S_semantic + (1 - α) × S_structural
                best_t, best_alpha, best_f1, best_M = find_best_combined_alignment(
                    G, S_semantic, S_structural
                )
            elif args.alignment == "smith-waterman":
                # TODO: penalties are hardcoded as a test.
                best_M, *_ = smith_waterman_align_affine_gap(
                    S_semantic + S_structural, -0.5, -0.01, 0.1
                )
                best_t = 0.0  # TODO
                best_alpha = 0.0  # TODO
                best_f1 = precision_recall_fscore_support(
                    G.flatten(), best_M.flatten(), average="binary", zero_division=0.0
                )[2]
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
                M_season = best_M[start:end, :]

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
                label=similarity_function,
            )

        ax.legend()
        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    else:
        raise ValueError(f"unknown alignment method: {args.similarity}")
