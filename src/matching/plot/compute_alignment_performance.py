# Compute several variations of alignment and output the
# resulting performance table
#
#
# Example usage:
#
# python compute_alignment_performance.py -m comics-novels -a structural -f plain
#
#
# For more details, see:
#
# python compute_structural_alignment_performance.py --help
#
#
# Author: Arthur Amalvy
from typing import List, Literal
import os, argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    load_medias_gold_alignment,
    load_medias_graphs,
    graph_similarity_matrix,
    get_episode_i,
    semantic_similarity,
    threshold_align_blocks,
    combined_similarities,
    load_medias_summaries,
    MEDIAS_STRUCTURAL_THRESHOLD,
    MEDIAS_SEMANTIC_THRESHOLD,
    MEDIAS_COMBINED_THRESHOLD,
)
from matching.plot.smith_waterman import (
    smith_waterman_align_affine_gap,
    MEDIAS_SMITH_WATERMAN_STRUCTURAL_PARAMS,
    MEDIAS_SMITH_WATERMAN_SEMANTIC_PARAMS,
    MEDIAS_SMITH_WATERMAN_COMBINED_PARAMS,
)


script_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--medias",
        type=str,
        help="Medias on which to compute alignment. Either 'comics-novels', 'tvshow-comics' or 'tvshow-novels'",
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
                            M = threshold_align_blocks(
                                S,
                                MEDIAS_STRUCTURAL_THRESHOLD[args.medias],
                                block_to_episode,
                            )
                        else:
                            M = S > MEDIAS_STRUCTURAL_THRESHOLD[args.medias]

                    elif args.alignment == "smith-waterman":
                        if args.blocks:
                            raise RuntimeError("unimplemented")

                        M, *_ = smith_waterman_align_affine_gap(
                            S, **MEDIAS_SMITH_WATERMAN_STRUCTURAL_PARAMS[args.medias]
                        )

                    else:
                        raise ValueError(f"unknown alignment method: {args.alignment}")

                    f1 = precision_recall_fscore_support(
                        G.flatten(),
                        M.flatten(),
                        average="binary",
                        zero_division=0.0,
                    )[2]

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
        assert not args.blocks

        first_summaries, second_summaries = load_medias_summaries(
            args.medias,
            args.min_delimiter_first_media,
            args.max_delimiter_first_media,
            args.min_delimiter_second_media,
            args.max_delimiter_second_media,
        )

        sim_fns: List[Literal["tfidf", "sbert"]] = ["tfidf", "sbert"]
        f1s = []
        for similarity_function in sim_fns:
            S = semantic_similarity(
                first_summaries, second_summaries, similarity_function
            )
            if args.alignment == "threshold":
                M = S > MEDIAS_SEMANTIC_THRESHOLD[args.medias][similarity_function]
            elif args.alignment == "smith-waterman":
                M, *_ = smith_waterman_align_affine_gap(
                    S,
                    **MEDIAS_SMITH_WATERMAN_SEMANTIC_PARAMS[args.medias][
                        similarity_function
                    ],
                )
            else:
                raise ValueError(f"unknown alignment method: {args.alignment}")

            f1 = precision_recall_fscore_support(
                G.flatten(), M.flatten(), average="binary", zero_division=0.0
            )[2]
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

        S_structural = graph_similarity_matrix(
            tvshow_graphs, novels_graphs, "edges", True
        )

        sim_fns: List[Literal["tfidf", "sbert"]] = ["tfidf", "sbert"]
        f1s = []

        for sim_fn in sim_fns:
            S_semantic = semantic_similarity(first_summaries, second_summaries, sim_fn)
            S_combined = combined_similarities(S_structural, S_semantic)

            if args.alignment == "threshold":
                threshold = MEDIAS_COMBINED_THRESHOLD[args.medias][sim_fn]
                M = S_combined > threshold
            elif args.alignment == "smith-waterman":
                M, *_ = smith_waterman_align_affine_gap(
                    S_semantic + S_structural,
                    **MEDIAS_SMITH_WATERMAN_COMBINED_PARAMS[args.medias][sim_fn],
                )
            else:
                raise ValueError(f"unknown alignment method: {args.alignment}")

            f1 = precision_recall_fscore_support(
                G.flatten(), M.flatten(), average="binary", zero_division=0.0
            )[2]
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

    else:
        raise ValueError(f"unknown similarity: {args.similarity}")
