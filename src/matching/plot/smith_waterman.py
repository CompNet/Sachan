from typing import Tuple, Dict, Optional, List, Literal, cast
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from alignment_commons import (
    load_medias_gold_alignment,
    load_medias_graphs,
    load_medias_summaries,
    graph_similarity_matrix,
    textual_similarity,
    combined_similarities,
)


def xnp_max(x: np.ndarray) -> Tuple[Tuple[int, ...], float]:
    idx = np.argmax(x)
    idx = np.unravel_index(idx, x.shape)
    return idx, x[idx]  # type: ignore


def xnp_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _smith_waterman_compute_matrices(
    S: np.ndarray,
    gap_start_penalty: float,
    gap_cont_penalty: float,
    neg_th: float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[str, int, int], Tuple[str, int, int]]
]:
    S = (xnp_sigmoid((S - np.mean(S)) / np.std(S)) - neg_th) * 2 - 1

    n, m = S.shape

    M = np.zeros((n + 1, m + 1))

    X = np.zeros((n + 1, m + 1))
    X[0, :] = gap_start_penalty + np.array(
        [(i + 1) * gap_cont_penalty for i in range(m + 1)]
    )

    Y = np.zeros((n + 1, m + 1))
    Y[0, :] = gap_start_penalty + np.array(
        [(j + 1) * gap_cont_penalty for j in range(m + 1)]
    )

    parents = {}

    vgap_start_penalty = gap_start_penalty * m / n
    vgap_cont_penalty = gap_cont_penalty * m / n
    hgap_start_penalty = gap_start_penalty * n / m
    hgap_cont_penalty = gap_cont_penalty * n / m

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = S[i - 1][j - 1]
            M_scores = [
                (("M", i - 1, j - 1), match_score + M[i - 1][j - 1]),
                (("X", i - 1, j - 1), match_score + X[i - 1][j - 1]),
                (("Y", i - 1, j - 1), match_score + Y[i - 1][j - 1]),
            ]
            Mparent, Mscore = max(
                M_scores, key=lambda parent_and_score: parent_and_score[1]
            )
            M[i][j] = Mscore
            parents[("M", i, j)] = Mparent

            X_scores = [
                (("M", i, j - 1), M[i][j - 1] - hgap_start_penalty - hgap_cont_penalty),
                (("X", i, j - 1), X[i][j - 1] - hgap_cont_penalty),
                (("Y", i, j - 1), Y[i][j - 1] - hgap_start_penalty - hgap_cont_penalty),
            ]
            Xparent, Xscore = max(
                X_scores, key=lambda parent_and_score: parent_and_score[1]
            )
            X[i][j] = Xscore
            parents[("X", i, j)] = Xparent

            Y_scores = [
                (("M", i - 1, j), M[i - 1][j] - vgap_start_penalty - vgap_cont_penalty),
                (("X", i - 1, j), X[i - 1][j] - vgap_start_penalty - vgap_cont_penalty),
                (("Y", i - 1, j), Y[i - 1][j] - vgap_cont_penalty),
            ]
            Yparent, Yscore = max(
                Y_scores, key=lambda parent_and_score: parent_and_score[1]
            )
            Y[i][j] = Yscore
            parents[("Y", i, j)] = Yparent

    return M, X, Y, parents


def _smith_waterman_backtrack(
    M: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    parents: Dict[Tuple[str, int, int], Tuple[str, int, int]],
    visited_mask: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    n = M.shape[0] - 1
    m = M.shape[1] - 1

    A = np.zeros((n, m))

    new_visited_mask = np.copy(visited_mask)

    name_to_matrix = {"M": M, "X": X, "Y": Y}

    # start at the cell with the highest score of the 3 matrices
    banned_scores = np.zeros((n + 1, m + 1))
    banned_scores[visited_mask == 1] = float("-inf")
    possible_starts = [
        ((matrix_name,) + coords, score)
        for matrix_name, (coords, score) in [
            ("M", xnp_max(M + banned_scores)),
            ("X", xnp_max(X + banned_scores)),
            ("Y", xnp_max(Y + banned_scores)),
        ]
    ]
    cell_address, cell_score = max(possible_starts, key=lambda ps: ps[1])

    if cell_score <= 0:
        return None

    while cell_score > 0:
        matrix_name, i, j = cell_address
        if i == 0 or j == 0:
            break

        if visited_mask[i][j] == 1:
            break

        A[i - 1][j - 1] = 1

        # we ban the current line and the current column from
        # subsequent matching
        new_visited_mask[i, :] = 1
        new_visited_mask[:, j] = 1

        cell_address = parents[cell_address]

        matrix_name, i, j = cell_address
        cell_score = name_to_matrix[matrix_name][i][j]

    if not np.any(A):
        return None

    return A, new_visited_mask


def smith_waterman_align_affine_gap(
    S: np.ndarray,
    gap_start_penalty: float,
    gap_cont_penalty: float,
    neg_th: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, m = S.shape

    M, X, Y, parents = _smith_waterman_compute_matrices(
        S, gap_start_penalty, gap_cont_penalty, neg_th
    )

    A = np.zeros((n, m))
    visited_mask = np.zeros((n + 1, m + 1))
    while np.any(visited_mask == 0):
        backtrack_out = _smith_waterman_backtrack(M, X, Y, parents, visited_mask)
        if backtrack_out is None:
            break
        A_i, visited_mask = backtrack_out
        A += A_i

    return A, M, X, Y


def tune_smith_waterman_params(
    X_tune: List[np.ndarray],
    G_tune: List[np.ndarray],
    gap_start_penalty_search_space: np.ndarray,
    gap_cont_penalty_search_space: np.ndarray,
    neg_th_search_space: np.ndarray,
    silent: bool = False,
) -> Tuple[float, float, float]:
    """Tune the parameters of smith-waterman using bruteforce.

    .. hint::

        Use :func:`np.arange` to specify search spaces.

    :param X_tune: list of similarity matrices
    :param G_tune: list of gold alignment matrices, one per similarity
        matrix.
    :param gap_start_penalty_search_space: search space for
        ``gap_start_penalty``
    :param gap_cont_penalty_search_space: search space for
        ``gap_cont_penalty``
    :param neg_th: search space for ``neg_th``

    :return: ``(gap_start_penalty, gat_cont_penalty, neg_th)``
    """
    best_gap_start_penalty = 0.0
    best_gap_cont_penalty = 0.0
    best_neg_th = 0.0
    best_f1 = 0.0

    progress = tqdm(gap_start_penalty_search_space, disable=silent)

    for gap_start_penalty in progress:
        for gap_cont_penalty in gap_cont_penalty_search_space:
            for neg_th in neg_th_search_space:
                f1_list = []
                for X, G in zip(X_tune, G_tune):
                    A, *_ = smith_waterman_align_affine_gap(
                        X, gap_start_penalty, gap_cont_penalty, neg_th
                    )
                    f1 = precision_recall_fscore_support(
                        G.flatten(), A.flatten(), average="binary", zero_division=0.0
                    )[2]
                    f1_list.append(f1)

                f1_mean = sum(f1_list) / len(f1_list)
                progress.set_description(
                    f"tuning ({gap_start_penalty:.2f}, {gap_cont_penalty:.2f}, {neg_th:.2f}, {f1_mean:.3f})"
                )
                if f1_mean > best_f1:
                    best_gap_start_penalty = gap_start_penalty
                    best_gap_cont_penalty = gap_cont_penalty
                    best_neg_th = neg_th
                    best_f1 = f1_mean

    return best_gap_start_penalty, best_gap_cont_penalty, best_neg_th


def tune_smith_waterman_params_other_medias(
    media_pair: Literal["tvshow-novels", "comics-novels", "tvshow-comics"],
    sim_mode: Literal["structural", "textual", "combined"],
    gap_start_penalty_search_space: np.ndarray,
    gap_cont_penalty_search_space: np.ndarray,
    neg_th_search_space: np.ndarray,
    textual_sim_fn: Literal["tfidf", "sbert"] = "tfidf",
    structural_mode: Literal["nodes", "edges"] = "edges",
    structural_use_weights: bool = True,
    structural_filtering: Literal["none", "common", "named", "common+named"] = "common",
    silent: bool = False,
) -> Tuple[float, float, float]:
    """
    Utility function, providing an higher level interface to
    :func:`tune_smith_waterman_params` by calling it correctly on the
    two other pair of medias beside ``media_pair``.

    :param media_pair: current media pairs.  Tuning will be performed
        on the two other media pairs.
    :param sim_mode: similarity measure, either ``'structural'`` or
        ``'textual'``
    :param gap_start_penalty_search_space: as in
        :func:`tune_smith_waterman_params`
    :param gap_cont_penalty_search_space: as in
        :func:`tune_smith_waterman_params`
    :param neg_th_search_space: as in
        :func:`tune_smith_waterman_params`
    :param textual_sim_fn: if ``sim_mode == 'textual'``, specifiy
        the similarity function (either ``'tfidf'`` or ``'sbert'``)
    :param structural_mode:
    :param structural_use_weights:
    :param structural_filtering:

    :return: ``(gap_start_penalty, gap_cont_penalty, neg_th)``
    """
    all_media_pairs = {"tvshow-novels", "comics-novels", "tvshow-comics"}
    other_media_pairs = all_media_pairs - {media_pair}

    X_tune = []
    G_tune = []

    for pair in other_media_pairs:
        pair = cast(Literal["tvshow-novels", "comics-novels", "tvshow-comics"], pair)

        G = load_medias_gold_alignment(pair)

        if sim_mode == "structural":
            first_media_graphs, second_media_graphs = load_medias_graphs(pair)
            X = graph_similarity_matrix(
                first_media_graphs,
                second_media_graphs,
                structural_mode,
                structural_use_weights,
                structural_filtering,
                silent=silent,
            )
        elif sim_mode == "textual":
            first_summaries, second_summaries = load_medias_summaries(pair)
            X = textual_similarity(
                first_summaries, second_summaries, textual_sim_fn, silent=silent
            )
        elif sim_mode == "combined":
            first_media_graphs, second_media_graphs = load_medias_graphs(pair)
            S_structural = graph_similarity_matrix(
                first_media_graphs,
                second_media_graphs,
                structural_mode,
                structural_use_weights,
                structural_filtering,
                silent=silent,
            )
            first_summaries, second_summaries = load_medias_summaries(pair)
            S_textual = textual_similarity(
                first_summaries, second_summaries, textual_sim_fn, silent=silent
            )
            X = combined_similarities(S_structural, S_textual)
        else:
            raise ValueError(f"unknown sim_mode: {sim_mode}")

        X = X[: G.shape[0], : G.shape[1]]
        X_tune.append(X)
        G_tune.append(G)

    return tune_smith_waterman_params(
        X_tune,
        G_tune,
        gap_start_penalty_search_space,
        gap_cont_penalty_search_space,
        neg_th_search_space,
        silent=silent,
    )


def smith_waterman_align_blocks(
    S: np.ndarray, block_to_episode: np.ndarray, **sw_kwargs
) -> np.ndarray:
    M_align_blocks, *_ = smith_waterman_align_affine_gap(S, **sw_kwargs)

    _, uniq_start_i = np.unique(block_to_episode, return_index=True)
    splits = np.split(M_align_blocks, uniq_start_i[1:], axis=0)

    M = []
    for split in splits:
        M.append(np.any(split, axis=0))

    M = np.stack(M)

    return M
