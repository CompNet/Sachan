# Compute several variations of structural alignment and output the
# resulting performance table
#
#
# Example usage:
#
# python compute_structural_alignment_performance.py -m novels-comics -f plain
#
#
# For more details, see:
#
# python compute_structural_alignment_performance.py --help
#
#
# Author: Arthur Amalvy
import argparse, os
import pandas as pd
import numpy as np
from alignment_commons import (
    find_best_alignment,
    find_best_blocks_alignment,
    load_medias_gold_alignment,
    load_medias_graphs,
    graph_similarity_matrix,
    get_episode_i,
)


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

    first_media_graphs, second_media_graphs = load_medias_graphs(
        args.medias,
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
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

                if args.blocks:
                    assert args.medias.startswith("tvshow")
                    block_to_episode = np.array(
                        [get_episode_i(G) for G in first_media_graphs]
                    )
                    _, f1, _ = find_best_blocks_alignment(G, S, block_to_episode)

                else:
                    _, f1, _ = find_best_alignment(G, S)

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
