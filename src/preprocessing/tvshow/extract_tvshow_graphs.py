# Extract the TVShow networks from the repository of Jeffrey Lancaster
# (see https://github.com/jeffreylancaster/game-of-thrones).
#
# Some example usage (assuming the user cloned Jeffrey Lancaster's repo at ~/game-of-thrones):
#
# python extract_tvshow_graphs.py -g episode -j ~/game-of-thrones -r
# python extract_tvshow_graphs.py -g scene -j ~/game-of-thrones -r
# python extract_tvshow_graphs.py -g block -bm locations -j ~/game-of-thrones -r
# python extract_tvshow_graphs.py -g block -bm similarity -bmk '{"threshold": 0.1}' -j ~/game-of-thrones -r
#
# Author: Arthur Amalvy
# 04/2023
import os, sys, argparse, json
import networkx as nx
from extraction import (
    load_characters_csv,
    load_got_tvshow_graphs,
    load_tvshow_character_map,
)
from graph_utils import cumulative_graph, relabeled_with_id
from tqdm import tqdm


script_dir = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
        help="one of 'episode', 'scene' or 'block'",
        default="scene",
    )
    parser.add_argument(
        "-bm",
        "--block-method",
        type=str,
        help="one of 'locations', 'similarity'.",
        default="locations",
    )
    parser.add_argument(
        "-bmk",
        "--block-method-kwargs",
        type=str,
        help="additional kwargs for the specified block method, as a json string. 'similarity' takes a 'threshold' argument between 0.0 and 1.0",
        default="{}",
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
        "-a",
        "--characters-csv-path",
        type=str,
        help="path to the characters.csv file",
        default=os.path.join(script_dir, "../../../in/characters.csv"),
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
        granularity = (
            f"block_{args.block_method}"
            if args.granularity == "block"
            else args.granularity
        )
        args.output_directory = os.path.join(
            script_dir, f"../../../in/tvshow/{iscumulative_dir}/{granularity}"
        )
    print(f"output dir: {args.output_directory}", file=sys.stderr)

    # Extract graphs from Jeffrey Lancaster repository
    charmap = load_tvshow_character_map(args.charmap_path)
    graphs = load_got_tvshow_graphs(
        args.jeffrey_lancaster_repo_path,
        args.granularity,
        charmap,
        block_method=args.block_method,
        block_method_kwargs=json.loads(args.block_method_kwargs),
    )
    graphs_len = len(graphs)
    # Exception: some characters have an incorrect name in the graphs
    mapping_exceptions = {"Olenna Tyrell": "Olenna Redwyne"}
    graphs = [nx.relabel_nodes(G, mapping_exceptions) for G in graphs]

    # Add the 'named' attribute for characters by using the
    # characters.csv file
    characters_df = load_characters_csv(args.characters_csv_path)
    # Exceptions: some characters are *not* in the characters.csv
    #             file, but should be named anyway
    missing_exceptions = ["Lord Blackmont", "Lord Portan"]
    # by default, all characters will have a "False" named attribute
    for G in graphs:
        for node, data in G.nodes(data=True):
            data["named"] = node in missing_exceptions
    # all characters found in characters.csv will get their named
    # attribute from there
    for _, line in characters_df.iterrows():
        for G in graphs:
            if line["Name"] in G.nodes:
                G.nodes[line["Name"]]["named"] = line["Named"]

    # Convert graphs to cumulative graphs if needed
    if args.cumulative:
        graphs = cumulative_graph(graphs)

    # Export graphs
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
