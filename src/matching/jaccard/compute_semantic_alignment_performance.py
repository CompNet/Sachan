# Plot the semantic alignment performance between the novels and the TV show
#
#
# Example usage:
#
# python compute_semantic_alignment_performance.py\
# --chapter-summaries './chapter_summaries.txt'\
# --episode-summaries './episodes_summaries.txt'\
# --similarity-function sbert
#
#
# For more details, see:
#
# python compute_structural_alignment_performance.py --help
#
#
# Author: Arthur Amalvy
import argparse, pickle
from typing import List, Literal
import pandas as pd
from plot_alignment_commons import find_best_alignment, semantic_similarity


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    CHAPTERS_NB = 344
    EPISODES_NB = 60

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--chapter-summaries",
        type=str,
        help="Path to a file with chapter summaries",
    )
    parser.add_argument(
        "-e",
        "--episode-summaries",
        type=str,
        help="Path to a file with episode summaries",
    )
    parser.add_argument(
        "-g",
        "--gold-alignment",
        type=str,
        help="Path to the gold chapters/episodes alignment.",
    )
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.chapter_summaries) as f:
        chapter_summaries = f.read().split("\n\n")[:-1]
    assert len(chapter_summaries) == CHAPTERS_NB

    with open(args.episode_summaries) as f:
        episode_summaries = f.read().split("\n\n")
    episode_summaries = episode_summaries[:EPISODES_NB]
    assert len(episode_summaries) == EPISODES_NB

    with open(args.gold_alignment, "rb") as f:
        G = pickle.load(f)
    G = G[:EPISODES_NB, :CHAPTERS_NB]
    assert G.shape == (EPISODES_NB, CHAPTERS_NB)

    sim_fns: List[Literal["tfidf", "sbert"]] = ["tfidf", "sbert"]
    f1s = []
    for similarity_function in sim_fns:
        S = semantic_similarity(
            episode_summaries, chapter_summaries, similarity_function
        )
        _, f1, _ = find_best_alignment(G, S)
        f1s.append(f1)

    performance_df = pd.DataFrame(f1s, columns=["F1"], index=sim_fns)

    LaTeX_export = (
        performance_df.style.format(lambda v: "{:.2f}".format(v * 100))
        .highlight_max(props="bfseries: ;", axis=None)
        .to_latex(hrules=True)
    )
    print(LaTeX_export)
