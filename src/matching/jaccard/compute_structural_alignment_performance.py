# Compute several variations of structural alignment and output the
# resulting performance as a LaTeX table
#
#
# Example usage:
#
# python compute_structural_alignment_performance.py --gold-alignment ./tvshow_novels_gold_alignment.pickle
#
#
# For more details, see:
#
# python compute_structural_alignment_performance.py --help
#
#
# Author: Arthur Amalvy
import argparse, pickle, sys, os
import pandas as pd
import numpy as np
from plot_alignment_commons import (
    NOVEL_LIMITS,
    TVSHOW_SEASON_LIMITS,
    find_best_alignment,
    load_tvshow_graphs,
    load_novels_graphs,
    graph_similarity_matrix,
)


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gold-alignment",
        type=str,
        default=None,
        help="Path to the gold chapters/episodes alignment. Must be specified if --best-treshold is specified.",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="latex",
        help="Dataframe print format. Either 'latex' or 'plain' (default: 'latex')",
    )
    parser.add_argument("-ms", "--min-season", type=int, default=1)
    parser.add_argument("-xs", "--max-season", type=int, default=6)
    parser.add_argument("-mn", "--min-novel", type=int, default=1)
    parser.add_argument("-xn", "--max-novel", type=int, default=5)
    args = parser.parse_args()

    max_novel_chapters = NOVEL_LIMITS[args.max_novel - 1]
    min_novel_chapters = ([0] + NOVEL_LIMITS)[max(0, args.min_novel - 1)]
    CHAPTERS_NB = max_novel_chapters - min_novel_chapters

    max_tvshow_episode = TVSHOW_SEASON_LIMITS[args.max_season - 1]
    min_tvshow_episode = ([0] + TVSHOW_SEASON_LIMITS)[max(0, args.min_season - 1)]
    EPISODES_NB = max_tvshow_episode - min_tvshow_episode

    print(CHAPTERS_NB)
    print(EPISODES_NB)

    # Load gold alignment matrix
    with open(args.gold_alignment, "rb") as f:
        G = pickle.load(f)
    G = G[:EPISODES_NB, :CHAPTERS_NB]
    assert G.shape == (EPISODES_NB, CHAPTERS_NB)

    novels_graphs = load_novels_graphs(args.min_novel, args.max_novel)
    assert len(novels_graphs) == CHAPTERS_NB

    tvshow_graphs = load_tvshow_graphs(args.min_season, args.max_season)
    assert len(tvshow_graphs) == EPISODES_NB

    sim_modes = ("nodes", "edges")
    use_weights_modes = (False, True)
    character_filtering_modes = ("none", "common", "named", "common+named")
    f1s = []

    for sim_mode in sim_modes:

        for use_weights in use_weights_modes:

            cf_f1s = []

            for character_filtering in character_filtering_modes:

                S = graph_similarity_matrix(
                    tvshow_graphs,
                    novels_graphs,
                    sim_mode,
                    use_weights,
                    character_filtering,
                )

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
