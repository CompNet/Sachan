from typing import Tuple, Sequence, Dict, Optional
import numpy as np


def xnp_max(x: np.ndarray) -> Tuple[Tuple[int, ...], float]:
    idx = np.argmax(x)
    idx = np.unravel_index(idx, x.shape)
    return idx, x[idx]


def xnp_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def smith_waterman_align(
    seq1: Sequence, seq2: Sequence, S: np.ndarray, W: np.ndarray, neg_th: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param seq1: first sequence to align, of length n
    :param seq2: second sequence to align, of length m
    :param S: similarity matrix, of shape (n x m)
    :param W: gap penalty, of shape max(n, m)

    :return: two matrices: an alignment matrix, and the intermediary
             score matrix
    """
    S = (xnp_sigmoid(S - np.mean(S)) - neg_th) * 2 - 1

    W = np.flip(W)

    n = len(seq1)
    m = len(seq2)

    assert S.shape == (n, m)

    H = np.zeros((n + 1, m + 1))
    H[0] = 0
    H[:, 0] = 0

    parents = {}

    # Compute H
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            linear_score = H[i - 1][j - 1] + S[i - 1][j - 1]
            vgap_score = max(H[:i, j] - W[-i:])
            hgap_score = max(H[i, :j] - W[-j:])

            scores = [
                ((i - 1, j - 1), linear_score),
                ((i - 1, j), vgap_score),
                ((i, j - 1), hgap_score),
                (None, 0),
            ]
            coords, score = max(scores, key=lambda c_and_s: c_and_s[1])

            H[i][j] = score

            if not coords is None:
                parents[(i, j)] = coords

    # backtrack
    A = np.zeros(S.shape)
    path_node = np.unravel_index(np.argmax(H), H.shape)
    while H[path_node] != 0:
        i, j = path_node
        A[i - 1, j - 1] = 1
        path_node = parents[path_node]
    i, j = path_node
    A[i - i, j - 1] = 1

    return A, H


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
        [i * gap_cont_penalty for i in range(m + 1)]  # i + 1 ?
    )

    Y = np.zeros((n + 1, m + 1))
    Y[0, :] = gap_start_penalty + np.array([j * gap_cont_penalty for j in range(m + 1)])

    parents = {}

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
                (("M", i, j - 1), gap_start_penalty + gap_cont_penalty + M[i][j - 1]),
                (("X", i, j - 1), gap_cont_penalty + X[i][j - 1]),
                (("Y", i, j - 1), gap_start_penalty + gap_cont_penalty + Y[i][j - 1]),
            ]
            Xparent, Xscore = max(
                X_scores, key=lambda parent_and_score: parent_and_score[1]
            )
            X[i][j] = Xscore
            parents[("X", i, j)] = Xparent

            Y[i][j] = max(
                gap_start_penalty + gap_cont_penalty + M[i - 1][j],
                gap_start_penalty + gap_cont_penalty + X[i - 1][j],
                gap_cont_penalty + Y[i - 1][j],
            )
            Y_scores = [
                (("M", i - 1, j), gap_start_penalty + gap_cont_penalty + M[i - 1][j]),
                (("X", i - 1, j), gap_start_penalty + gap_cont_penalty + X[i - 1][j]),
                (("Y", i - 1, j), gap_cont_penalty + Y[i - 1][j]),
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

        if visited_mask[i][j] == 1:
            break

        A[i - 1][j - 1] = 1

        # we ban the current line and the current column from
        # subsequent matching
        new_visited_mask[i, :] = 1
        new_visited_mask[:, j] = 1

        try:
            cell_address = parents[cell_address]
        # we arrived at the end of the matrix: return
        except KeyError:
            break

        matrix_name, i, j = cell_address
        cell_score = name_to_matrix[matrix_name][i][j]

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
