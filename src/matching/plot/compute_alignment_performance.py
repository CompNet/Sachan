from typing import List, Literal
import argparse, pickle, os, itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    combined_similarities_blocks,
    load_medias_gold_alignment,
    load_medias_graphs,
    load_medias_summaries,
    graph_similarity_matrix,
    textual_similarity,
    tune_alpha_other_medias,
    tune_threshold_other_medias,
    combined_similarities,
    align_blocks,
)
from smith_waterman import (
    smith_waterman_align_affine_gap,
    smith_waterman_align_blocks,
    tune_smith_waterman_params_other_medias,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--medias",
        type=str,
        help="Medias on which to compute alignment. Either 'comics-novels', 'tvshow-comics' or 'tvshow-novels'",
    )
    parser.add_argument(
        "-p",
        "--period",
        type=str,
        default=None,
        help="either unspecified (default for media pair) or one of U2, U5",
    )
    parser.add_argument(
        "-c",
        "--cumulative",
        action="store_true",
        help="if specified, perform alignment using the cumulative networks (if applicable)",
    )
    parser.add_argument(
        "-s",
        "--similarity",
        type=str,
        default="structural",
        help="One of 'structural', 'textual' 'combined'",
    )
    parser.add_argument(
        "-b",
        "--blocks",
        action="store_true",
        help="If specified, will use blocks when performing narrative matching. This can only be used when 'similarity' is 'structural' and 'medias' is 'tvshow-novels' or 'comics-novels'. In the first case, automatically extract blocks from the tvshow to help matching with the novels chapters. In the second case, use comics issues chapters to help matching with the novels chapters.",
    )
    args = parser.parse_args()

    if args.blocks:
        assert args.similarity in ["structural", "combined"]

    if not args.period is None:
        if args.period == "U2":
            delimiters = (1, 2, 1, 2)
        elif args.period == "U5":
            delimiter = (1, 5, 1, 5)
        else:
            raise ValueError(
                f"unknown period: {args.period}. Use no period, or one of U2, U5."
            )
    else:
        if args.medias == "tvshow-novels":
            delimiters = (1, 5, 1, 5)
        elif args.medias in ["tvshow-comics", "comics-novels"]:
            delimiters = (1, 2, 1, 2)
        else:
            raise ValueError(f"unknown media pair: {args.medias}")
    assert delimiters  # type: ignore

    G = load_medias_gold_alignment(args.medias, *delimiters)

    if args.similarity == "structural":
        first_media_graphs, second_media_graphs = load_medias_graphs(
            args.medias,
            *delimiters,
            tvshow_blocks="locations",
            comics_blocks=bool(args.blocks),
            cumulative=args.cumulative,
        )

        sim_modes: List[Literal["nodes", "edges"]] = ["nodes", "edges"]
        use_weights_modes = (False, True)
        character_filtering_modes: List[Literal["named", "common", "top20"]] = [
            "named",
            "common",
            "top20",
        ]

        columns = [
            "sim_mode",
            "use_weights",
            "character_filtering",
            "alignment",
            "f1",
            "precision",
            "recall",
        ]
        metrics_lst = []

        with tqdm(
            total=len(sim_modes)
            * len(use_weights_modes)
            * len(character_filtering_modes)
        ) as pbar:
            for sim_mode in sim_modes:
                for use_weights in use_weights_modes:
                    for character_filtering in character_filtering_modes:
                        if character_filtering == "top20":
                            if delimiters == (1, 2, 1, 2):
                                character_filtering = "top20s2"
                            elif delimiters == (1, 5, 1, 5):
                                character_filtering = "top20s5"
                            else:
                                raise ValueError(
                                    f"impossible delimiters/filtering combo ({delimiters}/{character_filtering})"
                                )

                        S = graph_similarity_matrix(
                            first_media_graphs,
                            second_media_graphs,
                            sim_mode,
                            use_weights,
                            character_filtering,
                            silent=True,
                        )

                        # threshold alignment
                        # -------------------
                        t = tune_threshold_other_medias(
                            args.medias,
                            "structural",
                            np.arange(0.0, 1.0, 0.01),
                            structural_mode=sim_mode,
                            structural_use_weights=use_weights,
                            structural_filtering=character_filtering,
                            silent=True,
                        )
                        M = S > t
                        if args.blocks:
                            M = align_blocks(
                                args.medias, first_media_graphs, second_media_graphs, M
                            )

                        precision, recall, f1, _ = precision_recall_fscore_support(
                            G.flatten(),
                            M.flatten(),
                            average="binary",
                            zero_division=0.0,
                        )

                        metrics_lst.append(
                            (
                                sim_mode,
                                use_weights,
                                character_filtering,
                                "threshold",
                                f1,
                                precision,
                                recall,
                            )
                        )

                        # SW alignment
                        # ------------
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
                            structural_mode=sim_mode,
                            structural_use_weights=use_weights,
                            structural_filtering=character_filtering,
                            silent=True,
                        )
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
                                S,
                                gap_start_penalty=gap_start_penalty,
                                gap_cont_penalty=gap_cont_penalty,
                                neg_th=neg_th,
                            )
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            G.flatten(),
                            M.flatten(),
                            average="binary",
                            zero_division=0.0,
                        )

                        metrics_lst.append(
                            (
                                sim_mode,
                                use_weights,
                                character_filtering,
                                "smith-waterman",
                                f1,
                                precision,
                                recall,
                            )
                        )

                        # tqdm update
                        # -----------
                        pbar.update(1)

    elif args.similarity == "textual":
        first_summaries, second_summaries = load_medias_summaries(
            args.medias, *delimiters
        )

        sim_fns: List[Literal["tfidf", "sbert"]] = ["tfidf", "sbert"]

        columns = ["sim_fn", "alignment", "f1", "precision", "recall"]
        metrics_lst = []

        for similarity_function in tqdm(sim_fns):
            S = textual_similarity(
                first_summaries, second_summaries, similarity_function, silent=True
            )

            # threshold alignment
            # -------------------
            t = tune_threshold_other_medias(
                args.medias,
                "textual",
                np.arange(0.0, 1.0, 0.01),
                textual_sim_fn=similarity_function,
                silent=True,
            )
            M = S > t

            precision, recall, f1, _ = precision_recall_fscore_support(
                G.flatten(), M.flatten(), average="binary", zero_division=0.0
            )
            metrics_lst.append(
                (similarity_function, "threshold", f1, precision, recall)
            )

            # SW alignment
            # ------------
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
                textual_sim_fn=similarity_function,
                silent=True,
            )
            M, *_ = smith_waterman_align_affine_gap(
                S,
                gap_start_penalty=gap_start_penalty,
                gap_cont_penalty=gap_cont_penalty,
                neg_th=neg_th,
            )

            precision, recall, f1, _ = precision_recall_fscore_support(
                G.flatten(), M.flatten(), average="binary", zero_division=0.0
            )
            metrics_lst.append(
                (similarity_function, "smith-waterman", f1, precision, recall)
            )

    elif args.similarity == "combined":
        columns = [
            "textual_sim_fn",
            "structural_sim_mode",
            "structural_use_weights",
            "structural_character_filtering",
            "alignment",
            "f1",
            "precision",
            "recall",
            "alpha",
        ]
        metrics_lst = []

        # sim_fn * mode * use_weights * filtering
        with tqdm(total=2 * 2 * 2 * 3) as pbar:
            first_graphs, second_graphs = load_medias_graphs(
                args.medias,
                *delimiters,
                cumulative=args.cumulative,
                tvshow_blocks="locations" if args.blocks else None,
                comics_blocks=args.blocks,
            )
            G = load_medias_gold_alignment(args.medias, *delimiters)

            first_summaries, second_summaries = load_medias_summaries(
                args.medias, *delimiters
            )

            sim_fn_lst: List[Literal["tfidf", "sbert"]] = ["tfidf", "sbert"]
            for sim_fn in sim_fn_lst:
                S_text = textual_similarity(
                    first_summaries, second_summaries, sim_fn, silent=True
                )

                modes: List[Literal["nodes", "edges"]] = ["nodes", "edges"]
                filtering_lst: List[Literal["named", "common", "top20"]] = [
                    "named",
                    "common",
                    "top20",
                ]
                for mode, use_weights, filtering in itertools.product(
                    modes, [True, False], filtering_lst
                ):
                    if filtering == "top20":
                        if delimiters == (1, 2, 1, 2):
                            filtering = "top20s2"
                        elif delimiters == (1, 5, 1, 5):
                            filtering = "top20s5"
                        else:
                            raise ValueError(
                                f"impossible delimiters/filtering combo ({delimiters}/{filtering})"
                            )

                    S_struct = graph_similarity_matrix(
                        first_graphs,
                        second_graphs,
                        mode,
                        use_weights,
                        filtering,
                        silent=True,
                    )

                    # threshold alignment
                    # -------------------
                    alpha, t = tune_alpha_other_medias(
                        args.medias,
                        "threshold",
                        np.arange(0.0, 1.0, 0.01),  # alpha
                        [np.arange(0.0, 1.0, 0.01)],  # threshold
                        sim_fn,
                        mode,
                        use_weights,
                        filtering,
                        silent=True,
                    )
                    if args.blocks:
                        S_combined = combined_similarities_blocks(
                            S_struct,
                            S_text,
                            alpha,
                            args.medias,
                            first_graphs,
                            second_graphs,
                        )
                        M = align_blocks(
                            args.medias,
                            first_graphs,
                            second_graphs,
                            S_combined > t,
                        )
                    else:
                        S_combined = combined_similarities(S_struct, S_text, alpha)
                        M = S_combined > t

                    precision, recall, f1, _ = precision_recall_fscore_support(
                        G.flatten(),
                        M.flatten(),
                        average="binary",
                        zero_division=0.0,
                    )

                    metrics_lst.append(
                        (
                            sim_fn,
                            mode,
                            use_weights,
                            filtering,
                            "threshold",
                            f1,
                            precision,
                            recall,
                        )
                    )

                    # SW alignment
                    # ------------
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
                        sim_fn,
                        mode,
                        use_weights,
                        filtering,
                        silent=True,
                    )
                    if args.blocks:
                        S_combined = combined_similarities_blocks(
                            S_struct,
                            S_text,
                            alpha,
                            args.medias,
                            first_graphs,
                            second_graphs,
                        )
                        M, *_ = smith_waterman_align_blocks(
                            args.medias,
                            first_graphs,
                            second_graphs,
                            S_combined,
                            gap_start_penalty=gap_start_penalty,
                            gap_cont_penalty=gap_cont_penalty,
                            neg_th=neg_th,
                        )
                    else:
                        S_combined = combined_similarities(S_struct, S_text, alpha)
                        M, *_ = smith_waterman_align_affine_gap(
                            S_combined, gap_start_penalty, gap_cont_penalty, neg_th
                        )

                    precision, recall, f1, _ = precision_recall_fscore_support(
                        G.flatten(),
                        M.flatten(),
                        average="binary",
                        zero_division=0.0,
                    )

                    metrics_lst.append(
                        (
                            sim_fn,
                            mode,
                            use_weights,
                            filtering,
                            "smith-waterman",
                            f1,
                            precision,
                            recall,
                            alpha,
                        )
                    )

                    pbar.update(1)

    else:
        raise ValueError(f"unknow similarity: {args.similarity}")

    df = pd.DataFrame(metrics_lst, columns=columns)
    dir_name = f"{args.medias}_{args.similarity}"
    if args.blocks:
        dir_name += "_blocks"
    if args.medias == "tvshow-novels" and args.period == "U2":
        dir_name += "_U2"
    if args.cumulative:
        dir_name += "_cumulative"
    dir_path = f"{root_dir}/out/matching/plot/{dir_name}"
    print(f"exporting results to {dir_path}...")
    os.makedirs(dir_path, exist_ok=True)
    with open(f"{dir_path}/df.pickle", "wb") as f:
        pickle.dump(df, f)
