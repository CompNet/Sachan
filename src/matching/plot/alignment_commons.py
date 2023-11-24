# Common functions for temporal matching
#
# Author: Arthur Amalvy
import pickle
from typing import Optional, Literal, List, Tuple
import copy, os, sys, glob
import numpy as np
import networkx as nx
from more_itertools import flatten
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize


# the end of each novel, in number of chapters
NOVEL_LIMITS = [73, 143, 225, 271, 344]

# the end of each season, in number of episodes
TVSHOW_SEASON_LIMITS = [10, 20, 30, 40, 50, 60, 67, 73]


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")


def load_tvshow_graphs(
    min_season: int = 1,
    max_season: int = 8,
    blocks: Optional[Literal["similarity", "locations"]] = None,
) -> List[nx.Graph]:
    """
    Load a graph for each episode of the TV show, from season
    ``min_season`` to ``max_season`` (inclusive)
    """
    tvshow_graphs = []
    if not blocks is None:
        paths = sorted(
            glob.glob(f"{root_dir}/in/tvshow/instant/block_{blocks}/*.graphml")
        )
    else:
        paths = sorted(glob.glob(f"{root_dir}/in/tvshow/instant/episode/*.graphml"))
    for path in paths:
        tvshow_graphs.append(nx.read_graphml(path))

    # only keep graphs between min_season and max_season
    tvshow_graphs = [G for G in tvshow_graphs if G.graph["season"] >= min_season]
    tvshow_graphs = [G for G in tvshow_graphs if G.graph["season"] <= max_season]

    # relabeling
    tvshow_graphs = [
        nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
        for G in tvshow_graphs
    ]

    return tvshow_graphs


def load_novels_graphs(min_novel: int = 1, max_novel: int = 5) -> List[nx.Graph]:
    """
    Load graphs for each chapter of the novels.

    :param min_novel: starting at 1, inclusive
    :param max_novel: up to 5, inclusive
    """
    ch_start = ([0] + NOVEL_LIMITS)[max(0, min_novel - 1)]
    ch_end = NOVEL_LIMITS[max_novel - 1]

    novel_graphs = []
    for path in sorted(glob.glob(f"{root_dir}/in/novels/instant/*.graphml"))[
        ch_start:ch_end
    ]:
        novel_graphs.append(nx.read_graphml(path))

    novel_graphs = [
        nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
        for G in novel_graphs
    ]

    return novel_graphs


def load_comics_graphs() -> List[nx.Graph]:
    """Load graphs for each chapter of the comics."""
    comics_graphs = []
    for path in sorted(glob.glob(f"{root_dir}/in/comics/instant/chapter/*.graphml")):
        comics_graphs.append(nx.read_graphml(path))

    comics_graphs = [
        nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
        for G in comics_graphs
    ]

    return comics_graphs


def load_medias_gold_alignment(
    medias: Literal["novels-comics", "tvshow-comics", "tvshow-novels"],
    min_delimiter_first_media: Optional[int] = None,
    max_delimiter_first_media: Optional[int] = None,
    min_delimiter_second_media: Optional[int] = None,
    max_delimiter_second_media: Optional[int] = None,
) -> np.ndarray:
    MEDIAS_TO_PATH = {
        "novels-comics": f"{root_dir}/in/plot_alignment/novels_comics_gold_alignment.pickle",
        "tvshow-comics": f"{root_dir}/in/plot_alignment/tvshow_comics_gold_alignment.pickle",
        "tvshow-novels": f"{root_dir}/in/plot_alignment/tvshow_novels_gold_alignment.pickle",
    }
    matrix_path = MEDIAS_TO_PATH.get(medias)
    if matrix_path is None:
        raise ValueError(f"wrong medias specification: {medias}")

    with open(matrix_path, "rb") as f:
        gold_matrix = pickle.load(f)

    if (
        medias == "tvshow-comics"
        and min_delimiter_first_media
        and max_delimiter_first_media
    ):
        ep_start = ([0] + TVSHOW_SEASON_LIMITS)[max(0, min_delimiter_first_media - 1)]
        ep_end = TVSHOW_SEASON_LIMITS[max_delimiter_first_media - 1]
        gold_matrix = gold_matrix[ep_start:ep_end, :]

    elif medias == "tvshow-novels":

        if min_delimiter_first_media is None:
            min_delimiter_first_media = 0
        if max_delimiter_first_media is None:
            max_delimiter_first_media = 8
        ep_start = ([0] + TVSHOW_SEASON_LIMITS)[max(0, min_delimiter_first_media - 1)]
        ep_end = TVSHOW_SEASON_LIMITS[max_delimiter_first_media - 1]

        if min_delimiter_second_media is None:
            min_delimiter_second_media = 0
        if max_delimiter_second_media is None:
            max_delimiter_second_media = 5
        ch_start = ([0] + NOVEL_LIMITS)[max(0, min_delimiter_second_media - 1)]
        ch_end = NOVEL_LIMITS[max_delimiter_second_media - 1]

        gold_matrix = gold_matrix[ep_start:ep_end, ch_start:ch_end]

    return gold_matrix


def load_medias_graphs(
    medias: Literal["novels-comics", "tvshow-comics", "tvshow-novels"],
    min_delimiter_first_media: Optional[int],
    max_delimiter_first_media: Optional[int],
    min_delimiter_second_media: Optional[int],
    max_delimiter_second_media: Optional[int],
) -> Tuple[nx.Graph, nx.Graph]:
    """Load the instant graphs for two medias to compare them.

    :return: (graph for first media, graph for second media)
    """
    splitted = medias.split("-")

    def load_graphs(
        media: str, min_delimiter: Optional[int], max_delimiter: Optional[int]
    ) -> List[nx.Graph]:
        if media == "novels":
            assert not min_delimiter is None
            assert not max_delimiter is None
            return load_novels_graphs(min_novel=min_delimiter, max_novel=max_delimiter)
        elif media == "comics":
            return load_comics_graphs()
        elif media == "tvshow":
            assert not min_delimiter is None
            assert not max_delimiter is None
            return load_tvshow_graphs(
                min_season=min_delimiter, max_season=max_delimiter
            )
        else:
            raise ValueError(f"wrong medias specification: {medias}")

    return (
        load_graphs(splitted[0], min_delimiter_first_media, max_delimiter_first_media),
        load_graphs(
            splitted[1], min_delimiter_second_media, max_delimiter_second_media
        ),
    )


def load_tvshow_episode_summaries(
    min_season: int = 1, max_season: int = 8
) -> List[str]:
    with open(f"{root_dir}/in/plot_alignment/tvshow_episode_summaries.txt") as f:
        episode_summaries = f.read().split("\n\n")
    ep_start = ([0] + TVSHOW_SEASON_LIMITS)[max(0, min_season - 1)]
    ep_end = TVSHOW_SEASON_LIMITS[max_season - 1]
    return episode_summaries[ep_start:ep_end]


def load_novels_chapter_summaries(min_novel: int = 1, max_novel: int = 5) -> List[str]:
    with open(f"{root_dir}/in/plot_alignment/novels_chapter_summaries.txt") as f:
        chapter_summaries = f.read().split("\n\n")[:-1]
    ch_start = ([0] + NOVEL_LIMITS)[max(0, min_novel - 1)]
    ch_end = NOVEL_LIMITS[max_novel - 1]
    return chapter_summaries[ch_start:ch_end]


def filtered_graph(G: nx.Graph, nodes: list) -> nx.Graph:
    """Return a graph where ``nodes`` are removed from ``G``"""
    H = copy.deepcopy(G)
    for node in nodes:
        if node in H:
            H.remove_node(node)
    return H


def get_episode_i(G: nx.Graph) -> int:
    """Get the index of the episode of ``G``

    .. note::

        only supports seasons from 1 to 6
    """
    assert G.graph["season"] < 7
    return (G.graph["season"] - 1) * 10 + G.graph["episode"] - 1


def jaccard_graph_sim(
    G: nx.Graph, H: nx.Graph, weights: Optional[dict], mode: Literal["nodes", "edges"]
) -> float:
    if mode == "nodes":
        g_set = set(G.nodes)
        h_set = set(H.nodes)
    elif mode == "edges":
        g_set = set([tuple(sorted(e)) for e in G.edges])
        h_set = set([tuple(sorted(e)) for e in H.edges])

    if not weights is None:
        inter = sum([weights.get(n, 0) for n in g_set.intersection(h_set)])
        union = sum([weights.get(n, 0) for n in g_set.union(h_set)])
    else:
        inter = len(g_set.intersection(h_set))
        union = len(g_set.union(h_set))

    if union == 0:
        return 0
    return inter / union


def graph_similarity_matrix(
    G_lst: List[nx.Graph],
    H_lst: List[nx.Graph],
    mode: Literal["nodes", "edges"],
    use_weights: bool,
    character_filtering: Literal["none", "common", "named", "common+named"] = "common",
) -> np.ndarray:
    """Compute a similarity matrixs between two lists of graph, using
    a tweaked Jaccard index.

    :param G_lst: a list of graphs of len ``n``
    :param H_lst: a list of graphs of len ``m``
    :param mode: Jaccard index compute methods

    :param use_weights: wether or not to weight Jaccard index by the
        number of occurences of:

            - Each character, if ``mode == "nodes"``

            - Interactions between characters, if ``mode == "edges"``

    :return: a similarity matrix of shape ``(n, m)``
    """
    # Character filtering
    G_chars = set(flatten([G.nodes for G in G_lst]))
    H_chars = set(flatten([H.nodes for H in H_lst]))

    if "named" in character_filtering:

        def is_named(char: str, graphs: List[nx.Graph]) -> bool:
            for J in graphs:
                attrs = J.nodes.get(char)
                if not attrs is None:
                    return attrs.get("named", attrs.get("Named"))
            return False

        G_unnamed_chars = [c for c in G_chars if not is_named(c, G_lst)]
        H_unnamed_chars = [c for c in H_chars if not is_named(c, H_lst)]
        G_lst = [filtered_graph(G, G_unnamed_chars) for G in G_lst]
        H_lst = [filtered_graph(H, H_unnamed_chars) for H in H_lst]

    if "common" in character_filtering:
        G_xor_H_chars = G_chars ^ H_chars
        G_lst = [filtered_graph(G, list(G_xor_H_chars)) for G in G_lst]
        H_lst = [filtered_graph(H, list(G_xor_H_chars)) for H in H_lst]

    # Nodes mode
    char_appearances = {}
    for G in G_lst + H_lst:
        for node in G.nodes:
            char_appearances[node] = char_appearances.get(node, 0) + 1

    # Edges mode
    rel_appearances = {}
    for G in G_lst + H_lst:
        for edge in G.edges:
            rel_appearances[tuple(sorted(edge))] = (
                rel_appearances.get(tuple(sorted(edge)), 0) + 1
            )

    # Compute n^2 similarity
    M = np.zeros((len(G_lst), len(H_lst)))

    for G_i, G in enumerate(tqdm(G_lst)):
        for H_i, H in enumerate(H_lst):
            weights = None
            if use_weights:
                weights = (
                    {c: 1 / n for c, n in char_appearances.items()}
                    if mode == "nodes"
                    else {c: 1 / n for c, n in rel_appearances.items()}
                )
            M[G_i][H_i] = jaccard_graph_sim(G, H, weights, mode=mode)

    return M


def semantic_similarity(
    episode_summaries: List[str],
    chapter_summaries: List[str],
    sim_fn: Literal["tfidf", "sbert"],
) -> np.ndarray:
    """Compute a semantic similarity matrix between episode and chapter summaries

    :return: a numpy array of shape (episodes_nb, chapters_nb)
    """
    episodes_nb = len(episode_summaries)
    chapters_nb = len(chapter_summaries)

    S = np.zeros((episodes_nb, chapters_nb))

    if sim_fn == "tfidf":

        vectorizer = TfidfVectorizer()
        v = vectorizer.fit(chapter_summaries + episode_summaries)

        chapters_v = vectorizer.transform(chapter_summaries)

        for i, e_summary in enumerate(tqdm(episode_summaries)):
            sents = sent_tokenize(e_summary)
            e_summary_v = vectorizer.transform(sents)
            chapter_sims = np.max(cosine_similarity(e_summary_v, chapters_v), axis=0)
            assert chapter_sims.shape == (chapters_nb,)
            S[i] = chapter_sims

    elif sim_fn == "sbert":

        print("Loading SentenceBERT model...", file=sys.stderr)
        stransformer = SentenceTransformer("all-mpnet-base-v2")

        print("Embedding chapter summaries...", file=sys.stderr)
        chapters_v = stransformer.encode(chapter_summaries)

        print(
            "Embedding episode summaries and computing similarity...", file=sys.stderr
        )
        for i, e_summary in enumerate(tqdm(episode_summaries)):
            sents = sent_tokenize(e_summary)
            e_summary_v = stransformer.encode(sents)
            chapters_sims = np.max(cosine_similarity(e_summary_v, chapters_v), axis=0)
            assert chapters_sims.shape == (chapters_nb,)
            S[i] = chapters_sims

    else:
        raise ValueError(
            f"Unknown similarity function: {sim_fn}. Use 'tfidf' or 'sbert'."
        )

    return S


def find_best_alignment(
    G: np.ndarray, S: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """Find the best possible alignment matrix by brute-force
    searching the best possible threshold.

    :param G: gold alignment matrix
    :param S: similarity matrix
    :return: ``(best threshold, best f1, best alignment matrix)``
    """
    best_t = 0.0
    best_f1 = 0.0
    best_M = S > 0.0
    for t in np.arange(0.0, 1.0, 0.01):
        M = S > t
        _, _, f1, _ = precision_recall_fscore_support(
            G.flatten(), M.flatten(), average="binary", zero_division=0.0
        )
        if f1 > best_f1:
            best_t = t
            best_f1 = f1
            best_M = M

    return (best_t, best_f1, best_M)


def find_best_blocks_alignment(
    G: np.ndarray, S: np.ndarray, block_to_episode: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """Given similarity between blocks and chapters, return the best
    mapping between chapters and episodes.

    :param G: gold alignment matrix ``(episodes_nb, chapters_nb)``
    :param S: ``(chapters_nb, blocks_nb)`` ``(blocks_nb, chapters_nb)``
    :param block_to_episode: ``(blocks_nb)``

    :return: ``(best threshold, best f1, best alignment matrix)``
    """
    best_t = 0.0
    best_f1 = 0.0
    best_M = S > 0.0

    for t in np.arange(0.0, 1.0, 0.01):

        M_align_blocks = S >= t

        _, uniq_start_i = np.unique(block_to_episode, return_index=True)
        splits = np.split(M_align_blocks, uniq_start_i[1:], axis=0)

        M = []
        for split in splits:
            M.append(np.any(split, axis=0))

        M = np.stack(M)

        _, _, f1, _ = precision_recall_fscore_support(
            G.flatten(), M.flatten(), average="binary", zero_division=0.0
        )
        if f1 > best_f1:
            best_t = t
            best_f1 = f1
            best_M = M

    return (best_t, best_f1, best_M)


def find_best_combined_alignment(
    G: np.ndarray, S_semantic: np.ndarray, S_structural: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """Find the best possible alignment matrix by brute-force
    searching the best possible threshold.

    :param G: gold alignment matrix
    :param S_semantic: semantic similarity matrix
    :param S_structural: structural similarity matrix
    :return: ``(best threshold, best alpha, best f1, best alignment matrix)``
    """
    alphas = np.arange(0.0, 1.0, 0.01)
    ts = np.arange(0.0, 1.0, 0.01)
    f1s = np.zeros((alphas.shape[0], ts.shape[0]))

    # Compute the best combination of both matrices
    # S_combined = α × S_semantic + (1 - α) × S_structural
    print("searching for α and t...", file=sys.stderr)
    for alpha_i, alpha in tqdm(enumerate(alphas), total=alphas.shape[0]):
        S = alpha * S_semantic + (1 - alpha) * S_structural
        for t_i, t in enumerate(ts):
            M = S > t
            _, _, f1, _ = precision_recall_fscore_support(
                G.flatten(),
                M.flatten(),
                average="binary",
                zero_division=0.0,
            )
            f1s[alpha_i][t_i] = f1

    best_f1_loc = np.argwhere(f1s == np.max(f1s))[0]
    best_f1 = float(np.max(f1s))
    best_alpha = float(best_f1_loc[0] / 100.0)
    best_t = float(best_f1_loc[1] / 100.0)
    best_S = best_alpha * S_semantic + (1 - best_alpha) * S_structural
    best_M = best_S > best_t

    return (best_t, best_alpha, best_f1, best_M)
