import argparse
import os
import networkx as nx
from got_extraction import load_got_book_graphs
from graph_utils import cumulative_graph


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-directory", type=str)
    parser.add_argument(
        "-s",
        "--csv-paths",
        nargs="+",
        type=str,
        help="paths of the books CSV",
    )
    parser.add_argument(
        "-c",
        "--cumulative",
        action="store_true",
        help="If specified, extract cumulative networks instead of instant networks",
    )
    args = parser.parse_args()

    graphs = []
    for book_path in args.csv_paths:
        graphs += load_got_book_graphs(book_path)
    if args.cumulative:
        graphs = cumulative_graph(graphs)

    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=True)
    for i, G in enumerate(graphs):
        output_file = os.path.join(output_directory, f"{i}.graphml")
        nx.write_graphml(G, output_file)
