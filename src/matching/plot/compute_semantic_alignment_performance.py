# Compute the semantic alignment performance between the novels and
# the TV show, and format it as a LaTeX table
#
#
# Example usage:
#
# python compute_semantic_alignment_performance.py
#
#
# For more details, see:
#
# python compute_semantic_alignment_performance.py --help
#
#
# Author: Arthur Amalvy
import argparse
from typing import List, Literal
import pandas as pd
from alignment_commons import (
    find_best_alignment,
    semantic_similarity,
    load_medias_gold_alignment,
    load_novels_chapter_summaries,
    load_tvshow_episode_summaries,
)


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--min-delimiter-first-media", type=int, default=1)
    parser.add_argument("-x1", "--max-delimiter-first-media", type=int, default=6)
    parser.add_argument("-m2", "--min-delimiter-second-media", type=int, default=1)
    parser.add_argument("-x2", "--max-delimiter-second-media", type=int, default=5)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    G = load_medias_gold_alignment(
        "tvshow-novels",
        min_delimiter_first_media=1,
        max_delimiter_first_media=6,
        min_delimiter_second_media=1,
        max_delimiter_second_media=5,
    )

    episode_summaries = load_tvshow_episode_summaries(
        args.min_delimiter_first_media, args.max_delimiter_first_media
    )
    chapter_summaries = load_novels_chapter_summaries(
        args.min_delimiter_second_media, args.max_delimiter_second_media
    )

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
