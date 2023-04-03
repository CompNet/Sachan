import argparse
import os
import networkx as nx
from got_extraction import load_got_tvshow_graphs
from graph_utils import cumulative_graph, relabeled_with_id


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-directory", type=str)
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

    graphs = load_got_tvshow_graphs(args.jeffrey_lancaster_repo_path, args.granularity)

    if args.relabel:
        graphs = [relabeled_with_id(G, "name") for G in graphs]

    if args.cumulative:
        graphs = cumulative_graph(graphs)

    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=True)
    for i, G in enumerate(graphs):
        output_file = os.path.join(output_directory, f"{i}.graphml")
        nx.write_graphml(G, output_file)
