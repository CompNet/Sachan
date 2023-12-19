from typing import Tuple, Sequence
import numpy as np


def xnp_max(x: np.ndarray) -> Tuple[Tuple[int, ...], float]:
    idx = np.argmax(x)
    idx = np.unravel_index(idx, x.shape)
    return idx, x[idx]


def smith_waterman_align(
    seq1: Sequence, seq2: Sequence, S: np.ndarray, W: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param seq1: first sequence to align, of length n
    :param seq2: second sequence to align, of length m
    :param S: similarity matrix, of shape (n x m)
    :param W: gap penalty, of shape max(n, m)

    :return: two matrices: an alignment matrix, and the intermediary
             score matrix
    """
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


def smith_waterman_align_affine_gap(
    x: Sequence,
    y: Sequence,
    S: np.ndarray,
    gap_start_penalty: float,
    gap_cont_penalty: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    m = len(y)
    assert S.shape == (n, m)

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

    # Backtracking
    # ------------
    A = np.zeros((n, m))

    name_to_matrix = {"M": M, "X": X, "Y": Y}

    # start at the cell with the highest score of the 3 matrices
    possible_starts = [
        ((matrix_name,) + coords, score)
        for matrix_name, (coords, score) in [
            ("M", xnp_max(M)),
            ("X", xnp_max(X)),
            ("Y", xnp_max(Y)),
        ]
    ]
    cell_address, cell_score = max(possible_starts, key=lambda ps: ps[1])
    while cell_score > 0:
        matrix_name, i, j = cell_address
        A[i - 1][j - 1] = 1
        cell_address = parents[cell_address]
        matrix_name, i, j = cell_address
        cell_score = name_to_matrix[matrix_name][i][j]
    _, i, j = cell_address
    A[i - 1][j - 1] = 1

    return A, M, X, Y
