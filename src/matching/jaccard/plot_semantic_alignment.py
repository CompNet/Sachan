# Plot the semantic alignment between the novels and the TV show
#
#
# Example usage:
#
# python plot_semantic_alignment.py\
# --chapter-summaries './chapter_summaries.txt'\
# --episode-summaries './episodes_summaries.txt'\
# --similarity-function sbert
#
#
# For more details, see:
#
# python plot_semantic_alignment.py --help
#
#
# Author: Arthur Amalvy
import argparse, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk import sent_tokenize


if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 3.0315

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
        "-s", "--similarity-function", type=str, help="Either 'tfidf' or 'sbert'."
    )
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    # Load summaries
    # --------------
    with open(args.chapter_summaries) as f:
        chapter_summaries = f.read().split("\n\n")[:-1]
    assert len(chapter_summaries) == CHAPTERS_NB

    with open(args.episode_summaries) as f:
        episode_summaries = f.read().split("\n\n")
    episode_summaries = episode_summaries[:EPISODES_NB]
    assert len(episode_summaries) == EPISODES_NB

    # Compute similarity
    # ------------------
    S = np.zeros((EPISODES_NB, CHAPTERS_NB))

    if args.similarity_function == "tfidf":

        vectorizer = TfidfVectorizer()
        v = vectorizer.fit(chapter_summaries + episode_summaries)

        chapters_v = vectorizer.transform(chapter_summaries)

        for i, e_summary in enumerate(tqdm(episode_summaries)):
            sents = sent_tokenize(e_summary)
            e_summary_v = vectorizer.transform(sents)
            chapter_sims = np.max(cosine_similarity(e_summary_v, chapters_v), axis=0)
            assert chapter_sims.shape == (CHAPTERS_NB,)
            S[i] = chapter_sims

    elif args.similarity_function == "sbert":

        print("loading SentenceBERT model...")
        stransformer = SentenceTransformer("all-mpnet-base-v2")

        print("Embedding chapter summaries...")
        chapters_v = stransformer.encode(chapter_summaries)

        print("Embedding episode summaries and computing similarity...")
        for i, e_summary in enumerate(tqdm(episode_summaries)):
            sents = sent_tokenize(e_summary)
            e_summary_v = stransformer.encode(sents)
            chapters_sims = np.max(cosine_similarity(e_summary_v, chapters_v), axis=0)
            assert chapters_sims.shape == (CHAPTERS_NB,)
            S[i] = chapters_sims

    else:
        raise ValueError(
            f"unknown similarity function: {args.similarity_function}. Use 'tfidf' or 'sbert'."
        )

    # Plot
    # ----
    fig, ax = plt.subplots()
    fig.set_size_inches(COLUMN_WIDTH_IN * 2, COLUMN_WIDTH_IN * 2 * 0.6)
    ax.imshow(S)
    ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
    ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
