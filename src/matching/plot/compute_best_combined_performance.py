#!/usr/bin/python3
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    combined_similarities,
    graph_similarity_matrix,
    load_medias_gold_alignment,
    load_medias_graphs,
    load_medias_summaries,
    semantic_similarity,
)
from smith_waterman import (
    tune_smith_waterman_params_other_medias,
    smith_waterman_align_affine_gap,
)


df_dict = {}

# media_pair * sim_fn * mode * use_weights * filtering
with tqdm(total=3 * 2 * 2 * 2 * 4) as pbar:

    for media_pair, media_delimiters in (
        ("tvshow-novels", (1, 6, 1, 5)),
        ("tvshow-comics", (1, 2, 1, 2)),
        ("comics-novels", (1, 2, 1, 2)),
    ):
        best_f1 = 0

        first_graphs, second_graphs = load_medias_graphs(media_pair, *media_delimiters)
        G = load_medias_gold_alignment(media_pair, *media_delimiters)

        first_summaries, second_summaries = load_medias_summaries(
            media_pair, *media_delimiters
        )

        for sim_fn in ("tfidf", "sbert"):

            S_sem = semantic_similarity(
                first_summaries, second_summaries, sim_fn, silent=True
            )

            for mode, use_weights, filtering in itertools.product(
                ["nodes", "edges"],
                [True, False],
                ["none", "common", "named", "common+named"],
            ):

                S_struct = graph_similarity_matrix(
                    first_graphs,
                    second_graphs,
                    mode,
                    use_weights,
                    filtering,
                    silent=True,
                )

                S_combined = combined_similarities(S_struct, S_sem)

                (
                    gap_start_penalty,
                    gap_cont_penalty,
                    neg_th,
                ) = tune_smith_waterman_params_other_medias(
                    media_pair,
                    "combined",
                    np.arange(0.0, 0.2, 0.01),
                    np.arange(0.0, 0.2, 0.01),
                    np.arange(0.0, 0.1, 0.1),  # effectively no search
                    sim_fn,
                    mode,
                    use_weights,
                    filtering,
                    silent=True,
                )

                M, *_ = smith_waterman_align_affine_gap(
                    S_combined, gap_start_penalty, gap_cont_penalty, neg_th
                )

                f1 = precision_recall_fscore_support(
                    G.flatten(), M.flatten(), average="binary", zero_division=0.0
                )[2]

                if f1 > best_f1:
                    best_f1 = f1

                pbar.update(1)

        df_dict[media_pair] = [best_f1]

df = pd.DataFrame.from_dict(df_dict)
LaTeX_export = (
    df.style.format(lambda v: "{:.2f}".format(v * 100))
    .hide(axis=0)  # hide id column
    .to_latex(hrules=True, column_format="ccc")
)
print(LaTeX_export)
