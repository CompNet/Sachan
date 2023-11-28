from typing import Tuple, Sequence
import numpy as np


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
