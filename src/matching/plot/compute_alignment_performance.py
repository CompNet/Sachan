from typing import List, Literal
import argparse, pickle, os, itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    get_comics_chapter_issue_i,
    load_medias_gold_alignment,
    load_medias_graphs,
    load_medias_summaries,
    graph_similarity_matrix,
    textual_similarity,
    tune_threshold_other_medias,
    combined_similarities,
    get_episode_i,
    threshold_align_blocks,
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

    # validate the use of args.blocks, since it is only relevant in
    # certain configurations
    if args.blocks:
        assert args.medias in ("tvhshow-novels", "comics-novels")
        assert args.similarity == "structural"

    if args.medias == "tvshow-novels":
        delimiters = (1, 5, 1, 5)
    elif args.medias == "tvshow-comics":
        delimiters = (1, 2, 1, 2)
    elif args.medias == "comics-novels":
        delimiters = (1, 2, 1, 2)
    else:
        raise ValueError(f"unknown media pair: {args.medias}")

    G = load_medias_gold_alignment(args.medias, *delimiters)

    if args.similarity == "structural":
        first_media_graphs, second_media_graphs = load_medias_graphs(
            args.medias,
            *delimiters,
            tvshow_blocks="locations" if args.blocks else None,
            comics_blocks=bool(args.blocks),
        )

        sim_modes: List[Literal["nodes", "edges"]] = ["nodes", "edges"]
        use_weights_modes = (False, True)
        character_filtering_modes: List[
            Literal["none", "common", "named", "common+named"]
        ] = ["none", "common", "named", "common+named"]

        columns = [
            "sim_mode",
            "use_weights",
            "character_filtering",
            "alignment",
            "f1",
            "precision",
            "recall",
        ]
        f1s = []

        with tqdm(
            total=len(sim_modes)
            * len(use_weights_modes)
            * len(character_filtering_modes)
        ) as pbar:
            for sim_mode in sim_modes:
                for use_weights in use_weights_modes:
                    for character_filtering in character_filtering_modes:
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
                        if args.blocks:
                            if args.medias == "tvshow-novels":
                                block_to_narrunit = np.array(
                                    [get_episode_i(G) for G in first_media_graphs]
                                )
                            else:
                                assert args.medias == "comics-novels"
                                block_to_narrunit = np.array(
                                    [
                                        get_comics_chapter_issue_i(G)
                                        for G in first_media_graphs
                                    ]
                                )
                            M = threshold_align_blocks(S, t, block_to_narrunit)
                        else:
                            M = S > t

                        precision, recall, f1, _ = precision_recall_fscore_support(
                            G.flatten(),
                            M.flatten(),
                            average="binary",
                            zero_division=0.0,
                        )

                        f1s.append(
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
                            # NOTE: we can 'type: ignore' on
                            # block_to_narrunit because we are sure it
                            # is computed when computing threshold
                            # alignment above if args.blocks is truthy
                            M = smith_waterman_align_blocks(
                                S,
                                block_to_narrunit,  # type: ignore
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

                        f1s.append(
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

        df = pd.DataFrame(f1s, columns=columns)
        dir_name = f"{root_dir}/out/matching/plot/{args.medias}_{args.similarity}"
        os.makedirs(dir_name, exist_ok=True)
        with open(f"{dir_name}/df.pickle", "wb") as f:
            pickle.dump(df, f)

    elif args.similarity == "textual":
        first_summaries, second_summaries = load_medias_summaries(
            args.medias, *delimiters
        )

        sim_fns: List[Literal["tfidf", "sbert"]] = ["tfidf", "sbert"]

        columns = ["sim_fn", "alignment", "f1", "precision", "recall"]
        f1s = []

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
            f1s.append((similarity_function, "threshold", f1, precision, recall))

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
            f1s.append((similarity_function, "smith-waterman", f1, precision, recall))

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
        ]
        f1s = []

        # sim_fn * mode * use_weights * filtering
        with tqdm(total=2 * 2 * 2 * 4) as pbar:
            first_graphs, second_graphs = load_medias_graphs(args.medias, *delimiters)
            G = load_medias_gold_alignment(args.medias, *delimiters)

            first_summaries, second_summaries = load_medias_summaries(
                args.medias, *delimiters
            )

            sim_fn_lst: List[Literal["tfidf", "sbert"]] = ["tfidf", "sbert"]
            for sim_fn in sim_fn_lst:
                S_sem = textual_similarity(
                    first_summaries, second_summaries, sim_fn, silent=True
                )

                modes: List[Literal["nodes", "edges"]] = ["nodes", "edges"]
                filtering_lst: List[
                    Literal["none", "common", "named", "common+named"]
                ] = ["none", "common", "named", "common+named"]
                for mode, use_weights, filtering in itertools.product(
                    modes, [True, False], filtering_lst
                ):
                    S_struct = graph_similarity_matrix(
                        first_graphs,
                        second_graphs,
                        mode,
                        use_weights,
                        filtering,
                        silent=True,
                    )

                    S_combined = combined_similarities(S_struct, S_sem)

                    # threshold alignment
                    # -------------------
                    t = tune_threshold_other_medias(
                        args.medias,
                        "combined",
                        np.arange(0.0, 1.0, 0.01),
                        textual_sim_fn=sim_fn,
                        structural_mode=mode,
                        structural_use_weights=use_weights,
                        structural_filtering=filtering,
                        silent=True,
                    )
                    M = S_combined > t

                    precision, recall, f1, _ = precision_recall_fscore_support(
                        G.flatten(),
                        M.flatten(),
                        average="binary",
                        zero_division=0.0,
                    )

                    f1s.append(
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
                        gap_start_penalty,
                        gap_cont_penalty,
                        neg_th,
                    ) = tune_smith_waterman_params_other_medias(
                        args.medias,
                        "combined",
                        np.arange(0.0, 0.2, 0.01),
                        np.arange(0.0, 0.2, 0.01),
                        np.arange(0.0, 0.1, 0.1),  # effectively no search
                        sim_fn,
                        mode,
                        use_weights,
                        filtering,
                        silent=True,
                    )

                    M, *_ = smith_waterman_align_affine_gap(
                        S_combined, gap_start_penalty, gap_cont_penalty, neg_th
                    )

                    precision, recall, f1, _ = precision_recall_fscore_support(
                        G.flatten(),
                        M.flatten(),
                        average="binary",
                        zero_division=0.0,
                    )

                    f1s.append(
                        (
                            sim_fn,
                            mode,
                            use_weights,
                            filtering,
                            "smith-waterman",
                            f1,
                            precision,
                            recall,
                        )
                    )

                    pbar.update(1)

    else:
        raise ValueError(f"unknow similarity: {args.similarity}")

    df = pd.DataFrame(f1s, columns=columns)
    dir_name = f"{root_dir}/out/matching/plot/{args.medias}_{args.similarity}"
    print(f"exporting results to {dir_name}...")
    os.makedirs(dir_name, exist_ok=True)
    with open(f"{dir_name}/df.pickle", "wb") as f:
        pickle.dump(df, f)
