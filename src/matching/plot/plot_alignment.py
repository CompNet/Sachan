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
    NOVEL_LIMITS,
    find_best_alignment,
    find_best_combined_alignment,
    graph_similarity_matrix,
    load_medias_gold_alignment,
    load_medias_graphs,
    get_episode_i,
    find_best_blocks_alignment,
    load_tvshow_episode_summaries,
    load_novels_chapter_summaries,
    semantic_similarity,
)
from smith_waterman import smith_waterman_align_affine_gap


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
                best_t, best_f1, best_S_align = find_best_blocks_alignment(
                    G, S, block_to_episode
                )
            else:
                best_t, best_f1, best_S_align = find_best_alignment(G, S)
        elif args.alignment == "smith-waterman":
            if args.blocks:
                raise RuntimeError("unimplemented")
            # TODO: penalties are hardcoded as a test.
            best_S_align, *_ = smith_waterman_align_affine_gap(S, -0.5, -0.01, 0.1)
            best_t = 0.0  # TODO
            best_f1 = precision_recall_fscore_support(
                G.flatten(), best_S_align.flatten(), average="binary", zero_division=0.0
            )[2]
        else:
            raise ValueError(f"unknown alignment method: {args.alignment}")

        print(f"{best_f1=}")
        print(f"{best_t=}")

        # Plot
        # ----
        plt.style.use("science")
        fig, ax = plt.subplots()
        fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)
        ax.set_title(f"t = {best_t}, F1 = {best_f1}", fontsize=FONTSIZE)
        ax.imshow(best_S_align, interpolation="none")
        ax.set_xlabel(args.medias.split("-")[0], fontsize=FONTSIZE)
        ax.set_ylabel(args.medias.split("-")[1], fontsize=FONTSIZE)

        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    elif args.similarity == "semantic":
        assert args.medias == "tvshow-novels"

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

        if args.alignment == "threshold":
            best_t, best_f1, best_S_align = find_best_alignment(G, S)
            print(f"{best_f1=}")
            print(f"{best_t=}")
        elif args.alignment == "smith-waterman":
            # TODO: penalties are hardcoded as a test.
            best_S_align, *_ = smith_waterman_align_affine_gap(S, -0.5, -0.01, 0.1)
            best_t = 0.0  # TODO
            best_f1 = precision_recall_fscore_support(
                G.flatten(), best_S_align.flatten(), average="binary", zero_division=0.0
            )[2]
        else:
            raise ValueError(f"unknown alignment method: {args.alignment}")

        # Plot
        # ----
        plt.style.use("science")
        fig, ax = plt.subplots()
        fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)
        ax.set_title(f"t = {best_t}, F1 = {best_f1}", fontsize=FONTSIZE)
        ax.imshow(best_S_align, interpolation="none")
        ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
        ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)

        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    elif args.similarity == "combined":
        assert args.medias == "tvshow-novels"
        assert not args.blocks

        # Load summaries
        # --------------
        episode_summaries = load_tvshow_episode_summaries(
            args.min_delimiter_first_media, args.max_delimiter_first_media
        )
        chapter_summaries = load_novels_chapter_summaries(
            args.min_delimiter_second_media, args.max_delimiter_second_media
        )

        # Load networks
        # -------------
        tvshow_graphs, novels_graphs = load_medias_graphs(
            "tvshow-novels",
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
            "locations" if args.blocks else None,
        )

        # Compute similarity
        # ------------------
        S_semantic = semantic_similarity(
            episode_summaries, chapter_summaries, args.similarity_function
        )
        S_structural = graph_similarity_matrix(
            tvshow_graphs, novels_graphs, "edges", True
        )

        # Combination
        # -----------
        if args.alignment == "threshold":
            best_t, best_alpha, best_f1, best_M = find_best_combined_alignment(
                G, S_semantic, S_structural
            )
        elif args.alignment == "smith-waterman":
            # TODO: penalty are hardcoded as a test.
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

        # Plot
        # ----
        plt.style.use("science")
        fig, ax = plt.subplots()
        fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)
        ax.set_title(
            f"alpha = {best_alpha}, t = {best_t}, F1 = {best_f1}", fontsize=FONTSIZE
        )
        ax.imshow(best_M, interpolation="none")
        ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
        ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)

        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, bbox_inches="tight")
        else:
            plt.show()

    else:
        raise ValueError(f"unknown alignment method: {args.alignment}")
