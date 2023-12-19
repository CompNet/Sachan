# Compute several variations of alignment and output the
# resulting performance table
#
#
# Example usage:
#
# python compute_alignment_performance.py -m novels-comics -a structural -f plain
#
#
# For more details, see:
#
# python compute_structural_alignment_performance.py --help
#
#
# Author: Arthur Amalvy
from typing import List, Literal
import argparse, os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    find_best_alignment,
    find_best_blocks_alignment,
    find_best_combined_alignment,
    load_medias_gold_alignment,
    load_medias_graphs,
    graph_similarity_matrix,
    get_episode_i,
    semantic_similarity,
    load_novels_chapter_summaries,
    load_tvshow_episode_summaries,
)
from matching.plot.smith_waterman import smith_waterman_align_affine_gap


script_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--medias",
        type=str,
        help="Medias on which to compute alignment. Either 'novels-comics', 'tvshow-comics' or 'tvshow-novels'",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="latex",
        help="Dataframe print format. Either 'latex' or 'plain' (default: 'latex')",
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
    args = parser.parse_args()

    G = load_medias_gold_alignment(
        args.medias,
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
        )

        sim_modes = ("nodes", "edges")
        use_weights_modes = (False, True)
        character_filtering_modes = ("none", "common", "named", "common+named")
        f1s = []

        for sim_mode in sim_modes:
            for use_weights in use_weights_modes:
                cf_f1s = []

                for character_filtering in character_filtering_modes:
                    S = graph_similarity_matrix(
                        first_media_graphs,
                        second_media_graphs,
                        sim_mode,  # type: ignore
                        use_weights,
                        character_filtering,  # type: ignore
                    )

                    if args.alignment == "threshold":

                        if args.blocks:
                            assert args.medias.startswith("tvshow")
                            block_to_episode = np.array(
                                [get_episode_i(G) for G in first_media_graphs]
                            )
                            _, f1, _ = find_best_blocks_alignment(
                                G, S, block_to_episode
                            )

                        else:
                            _, f1, _ = find_best_alignment(G, S)

                    elif args.alignment == "smith-waterman":

                        if args.blocks:
                            raise RuntimeError("unimplemented")

                        # TODO: penalty are hardcoded as a test. Dont forget to
                        # change them when S will be modified in the
                        # function. Penalty also depend on the similarity type.
                        M, *_ = smith_waterman_align_affine_gap(
                            first_media_graphs, second_media_graphs, S, 0.1, -0.01
                        )
                        f1 = precision_recall_fscore_support(
                            G.flatten(),
                            M.flatten(),
                            average="binary",
                            zero_division=0.0,
                        )[2]

                    else:
                        raise ValueError(f"unknown alignment method: {args.alignment}")

                    cf_f1s.append(f1)

                f1s.append(cf_f1s)

        use_weights_display = {False: "no", True: "yes"}
        mcolumns = pd.MultiIndex.from_product(
            [sim_modes, [use_weights_display[m] for m in use_weights_modes]],
            names=["Jaccard index", "weighted"],
        )

        performance_df = pd.DataFrame(
            np.array(f1s).T,
            index=character_filtering_modes,
            columns=mcolumns,
        )
        performance_df.index.name = "character filtering"

        if args.format == "latex":
            LaTeX_export = (
                performance_df.style.format(lambda v: "{:.2f}".format(v * 100))
                .highlight_max(props="bfseries: ;", axis=None)
                .to_latex(hrules=True, sparse_index=False, multicol_align="c")
            )
            print(LaTeX_export)
        else:
            print(performance_df)

    elif args.similarity == "semantic":
        assert args.medias == "tvshow-novels"
        assert not args.blocks

        episode_summaries = load_tvshow_episode_summaries(
            args.min_delimiter_first_media, args.max_delimiter_first_media
        )
        chapter_summaries = load_novels_chapter_summaries(
            args.min_delimiter_second_media, args.max_delimiter_second_media
        )

        sim_fns: List[Literal["tfidf", "sbert"]] = ["tfidf", "sbert"]
        f1s = []
        for similarity_function in sim_fns:
            S = semantic_similarity(
                episode_summaries, chapter_summaries, similarity_function
            )
            if args.alignment == "threshold":
                _, f1, _ = find_best_alignment(G, S)
            elif args.alignment == "smith-waterman":
                # TODO: penalty are hardcoded as a test. Dont forget to
                # change them when S will be modified in the
                # function. Penalty also depend on the similarity type.
                M, *_ = smith_waterman_align_affine_gap(
                    episode_summaries, chapter_summaries, S, -0.1, -0.01
                )
                f1 = precision_recall_fscore_support(
                    G.flatten(), M.flatten(), average="binary", zero_division=0.0
                )[2]
            else:
                raise ValueError(f"unknown alignment method: {args.alignment}")

            f1s.append(f1)

        performance_df = pd.DataFrame(f1s, columns=["F1"], index=sim_fns)

        if args.format == "latex":
            LaTeX_export = (
                performance_df.style.format(lambda v: "{:.2f}".format(v * 100))
                .highlight_max(props="bfseries: ;", axis=None)
                .to_latex(hrules=True)
            )
            print(LaTeX_export)
        else:
            print(performance_df)

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

        S_semantic = semantic_similarity(
            episode_summaries, chapter_summaries, args.similarity_function
        )

        S_structural = graph_similarity_matrix(
            tvshow_graphs, novels_graphs, "edges", True, "common"
        )

        # Combination
        # -----------
        if args.alignment == "threshold":
            best_t, best_alpha, best_f1, _ = find_best_combined_alignment(
                G, S_semantic, S_structural
            )
            print(f"{best_alpha=}")
            print(f"{best_t=}")
        elif args.alignment == "smith-waterman":
            # TODO: penalty are hardcoded as a test. Dont forget to
            # change them when S will be modified in the
            # function. Penalty also depend on the similarity type.
            best_M, *_ = smith_waterman_align_affine_gap(
                tvshow_graphs, novels_graphs, S_semantic + S_structural, -0.1, -0.01
            )
            best_f1 = precision_recall_fscore_support(
                G.flatten(), best_M.flatten(), average="binary", zero_division=0.0
            )[2]
        else:
            raise ValueError(f"unknown alignment method: {args.alignment}")
        print(f"{best_f1=}", flush=True)

    else:
        raise ValueError(f"unknown alignment method: {args.similarity}")
