import argparse
import os
from typing import Literal, cast
import networkx as nx
from got_extraction import (
    load_got_tvshow_conversational_scene_graphs,
    load_got_tvshow_conversational_episode_graphs,
)
from graph_utils import cumulative_graph


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
        "-r",
        "--graphml-path",
        type=str,
        help="path to the graphml file obtained using https://github.com/bostxavier/Narrative-Smoothing. Must be specified if --granularity='scene'",
    )
    parser.add_argument(
        "-t",
        "--got-json-path",
        type=str,
        help="path to the got.json file obtained using https://github.com/bostxavier/Narrative-Smoothing. Must be specified if --granularity='episode'",
    )
    parser.add_argument(
        "-c",
        "--cumulative",
        action="store_true",
        help="If specified, extract cumulative networks instead of instant networks",
    )
    args = parser.parse_args()

    if args.granularity == "scene":
        graphs = load_got_tvshow_conversational_scene_graphs(args.graphml_path)
    elif args.granularity == "episode":
        graphs = []
        for season in range(1, 9):
            season = cast(Literal[1, 2, 3, 4, 5, 6, 7, 8], season)
            graphs += load_got_tvshow_conversational_episode_graphs(
                args.got_json_path, season
            )
    else:
        raise ValueError

    if args.cumulative:
        graphs = list(cumulative_graph(graphs))

    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=True)
    for i, G in enumerate(graphs):
        output_file = os.path.join(output_directory, f"{i}.graphml")
        nx.write_graphml(G, output_file)
