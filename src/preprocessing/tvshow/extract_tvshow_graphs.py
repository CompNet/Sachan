# Extract the TVShow networks from the repository of Jeffrey Lancaster
# (see https://github.com/jeffreylancaster/game-of-thrones).
#
# Author: Arthur Amalvy
# 04/2023
import argparse
import os, sys
import networkx as nx
from extraction import load_got_tvshow_graphs, load_tvshow_character_map
from graph_utils import cumulative_graph, relabeled_with_id
from tqdm import tqdm


script_dir = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        help="output directory. Default will be in the appropriate directory in ../../../in/tvshow",
    )
    parser.add_argument(
        "-g",
        "--granularity",
        type=str,
        help="one of 'episode', 'scene'",
        default="scene",
    )
    parser.add_argument(
        "-j",
        "--jeffrey-lancaster-repo-path",
        type=str,
        help="path to Jeffrey Lancaster's game-of-thrones repo.",
    )
    parser.add_argument(
        "-m",
        "--charmap-path",
        type=str,
        help="path to the characters map csv",
        default=os.path.join(script_dir, "../../../in/tvshow/charmap.csv"),
    )
    parser.add_argument(
        "-c",
        "--cumulative",
        action="store_true",
        help="If specified, extract cumulative networks instead of instant networks",
    )
    parser.add_argument(
        "-r",
        "--relabel",
        action="store_true",
        help="If specified, relabel nodes with an ID instead of character names, and put names as node attributes to save disk space",
    )
    args = parser.parse_args()

    if args.output_directory is None:
        iscumulative_dir = "cumul" if args.cumulative else "instant"
        args.output_directory = os.path.join(
            script_dir, f"../../../in/tvshow/{iscumulative_dir}/{args.granularity}"
        )
    print(f"output dir: {args.output_directory}", file=sys.stderr)

    charmap = load_tvshow_character_map(args.charmap_path)
    graphs = load_got_tvshow_graphs(
        args.jeffrey_lancaster_repo_path, args.granularity, charmap
    )
    graphs_len = len(graphs)

    if args.cumulative:
        graphs = cumulative_graph(graphs)

    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=True)
    graph_i_str_len = len(str(graphs_len))
    for i, G in tqdm(enumerate(graphs), total=graphs_len):
        if args.relabel:
            G = relabeled_with_id(G, "name")
        iscumulative_str = "cumulative" if args.cumulative else "instant"
        graph_i_str = str(i).rjust(graph_i_str_len, "0")
        output_file = os.path.join(
            output_directory, f"{iscumulative_str}_{graph_i_str}.graphml"
        )
        nx.write_graphml(G, output_file)
