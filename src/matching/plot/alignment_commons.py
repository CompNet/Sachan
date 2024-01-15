# Common functions for temporal matching
#
# Author: Arthur Amalvy
from collections import defaultdict
from typing import Optional, Literal, List, Tuple, cast
import copy, os, sys, glob, pickle
import numpy as np
import pandas as pd
import networkx as nx
from more_itertools import flatten
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from graph_utils import cumulative_graph


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


def load_comics_graphs(
    granularity: Literal["issue", "chapter"] = "issue"
) -> List[nx.Graph]:
    """Load graphs for each chapter of the comics.

    :param granularity: either 'issue' (one graph per issue) or
        'chapter' (one graph per book chapter)
    """
    assert granularity in ("issue", "chapter")

    comics_graphs = []
    for path in sorted(glob.glob(f"{root_dir}/in/comics/instant/chapter/*.graphml")):
        comics_graphs.append(nx.read_graphml(path))

    comics_graphs = [
        nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
        for G in comics_graphs
    ]

    if granularity == "chapter":
        return comics_graphs

    # granularity == "issue"
    # comics_graphs has one graph per novel chapter. We must convert
    # that to one graph per issue (an issue contains several chapters
    # from the novel)
    df = pd.read_csv(f"{root_dir}/in/plot_alignment/comics_mapping.csv")
    chapter_to_issue = dict(zip(df["Rank"], df["Volume"]))  # example: {1: "AGOT01"}

    # group graphs by issue
    issue_graphs = defaultdict(list)
    for graph in comics_graphs:
        chapter_str = graph.graph["NarrUnit"]
        chapter = int(chapter_str.split("_")[1])
        issue = chapter_to_issue[chapter]
        issue_graphs[issue].append(graph)

    # join episode graphs for each issue
    issue_graphs = {k: list(cumulative_graph(v))[-1] for k, v in issue_graphs.items()}

    # sort issues correctly
    sorted_keys = sorted([k for k in issue_graphs if k.startswith("AGOT")])
    sorted_keys += sorted([k for k in issue_graphs if k.startswith("ACOK")])
    issue_graphs = [issue_graphs[issue] for issue in sorted_keys]

    # sanity check
    assert len(issue_graphs) == 56

    return issue_graphs


def load_medias_gold_alignment(
    medias: Literal["comics-novels", "tvshow-comics", "tvshow-novels"],
    min_delimiter_first_media: Optional[int] = None,
    max_delimiter_first_media: Optional[int] = None,
    min_delimiter_second_media: Optional[int] = None,
    max_delimiter_second_media: Optional[int] = None,
) -> np.ndarray:
    MEDIAS_TO_PATH = {
        "comics-novels": f"{root_dir}/in/plot_alignment/comics_novels_i2c_gold_alignment.pickle",
        "tvshow-comics": f"{root_dir}/in/plot_alignment/tvshow_comics_i2e_gold_alignment.pickle",
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
            min_delimiter_first_media = 1
        if max_delimiter_first_media is None:
            max_delimiter_first_media = 5
        ep_start = ([0] + TVSHOW_SEASON_LIMITS)[max(0, min_delimiter_first_media - 1)]
        ep_end = TVSHOW_SEASON_LIMITS[max_delimiter_first_media - 1]

        if min_delimiter_second_media is None:
            min_delimiter_second_media = 1
        if max_delimiter_second_media is None:
            max_delimiter_second_media = 5
        ch_start = ([0] + NOVEL_LIMITS)[max(0, min_delimiter_second_media - 1)]
        ch_end = NOVEL_LIMITS[max_delimiter_second_media - 1]

        gold_matrix = gold_matrix[ep_start:ep_end, ch_start:ch_end]

    return gold_matrix


def load_medias_graphs(
    medias: Literal["comics-novels", "tvshow-comics", "tvshow-novels"],
    min_delimiter_first_media: Optional[int] = None,
    max_delimiter_first_media: Optional[int] = None,
    min_delimiter_second_media: Optional[int] = None,
    max_delimiter_second_media: Optional[int] = None,
    tvshow_blocks: Optional[Literal["locations", "similarity"]] = None,
) -> Tuple[List[nx.Graph], List[nx.Graph]]:
    """Load the instant graphs for two medias to compare them.

    :return: (graph for first media, graph for second media)
    """
    first_media, second_media = medias.split("-")

    def load_graphs(
        media: str, min_delimiter: Optional[int], max_delimiter: Optional[int]
    ) -> List[nx.Graph]:
        if media == "novels":
            return load_novels_graphs(
                min_novel=min_delimiter or 1, max_novel=max_delimiter or 5
            )
        elif media == "comics":
            return load_comics_graphs()
        elif media == "tvshow":
            return load_tvshow_graphs(
                min_season=min_delimiter or 1,
                max_season=max_delimiter or 5,
                blocks=tvshow_blocks,
            )
        else:
            raise ValueError(f"wrong medias specification: {medias}")

    return (
        load_graphs(first_media, min_delimiter_first_media, max_delimiter_first_media),
        load_graphs(
            second_media, min_delimiter_second_media, max_delimiter_second_media
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


def load_comics_issue_summaries() -> List[str]:
    with open(f"{root_dir}/in/plot_alignment/comics_issue_summaries.txt") as f:
        comics_summaries = f.read().split("\n\n")
        comics_summaries = [
            s if not s == "NO SUMMARY" else "" for s in comics_summaries
        ]
    return comics_summaries


def load_medias_summaries(
    medias: Literal["comics-novels", "tvshow-comics", "tvshow-novels"],
    min_delimiter_first_media: Optional[int] = None,
    max_delimiter_first_media: Optional[int] = None,
    min_delimiter_second_media: Optional[int] = None,
    max_delimiter_second_media: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    first_media, second_media = medias.split("-")

    def load_summaries(
        media: str, min_delimiter: Optional[int], max_delimiter: Optional[int]
    ) -> List[str]:
        if media == "novels":
            return load_novels_chapter_summaries(
                min_novel=min_delimiter or 1, max_novel=max_delimiter or 5
            )
        elif media == "comics":
            return load_comics_issue_summaries()
        elif media == "tvshow":
            return load_tvshow_episode_summaries(
                min_season=min_delimiter or 1, max_season=max_delimiter or 5
            )
        else:
            raise ValueError(f"wrong medias specification: {medias}")

    return (
        load_summaries(
            first_media, min_delimiter_first_media, max_delimiter_first_media
        ),
        load_summaries(
            second_media, min_delimiter_second_media, max_delimiter_second_media
        ),
    )


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

        only supports seasons from 1 to 5
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
    silent: bool = False,
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

    :param silent: if False, displays a progress bar

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

    for G_i, G in enumerate(tqdm(G_lst, disable=silent)):
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


def textual_similarity(
    first_summaries: List[str],
    second_summaries: List[str],
    sim_fn: Literal["tfidf", "sbert"],
    silent: bool = False,
) -> np.ndarray:
    """Compute a textual similarity matrix between summaries

    :return: a numpy array of shape (len(first_summaries), len(second_summaries))
    """
    if sim_fn == "tfidf":
        vectorizer = TfidfVectorizer()
        _ = vectorizer.fit(first_summaries + second_summaries)

        first_v = vectorizer.transform(first_summaries)
        second_v = vectorizer.transform(second_summaries)

        S = cosine_similarity(first_v, second_v)

    elif sim_fn == "sbert":
        from sentence_transformers import SentenceTransformer

        if not silent:
            print("Loading SentenceBERT model...", file=sys.stderr)
        stransformer = SentenceTransformer("all-mpnet-base-v2")

        if not silent:
            print("Embedding first summaries...", file=sys.stderr)
        first_v = stransformer.encode(first_summaries)

        if not silent:
            print("Embedding second summaries...", file=sys.stderr)
        second_v = stransformer.encode(second_summaries)

        S = cosine_similarity(first_v, second_v)

    else:
        raise ValueError(
            f"Unknown similarity function: {sim_fn}. Use 'tfidf' or 'sbert'."
        )

    return S


def combined_similarities(
    S_structural: np.ndarray, S_textual: np.ndarray
) -> np.ndarray:
    S_textual = (S_textual - np.min(S_textual)) / (
        np.max(S_textual) - np.min(S_textual)
    )
    S_structural = (S_structural - np.min(S_structural)) / (
        np.max(S_structural) - np.min(S_structural)
    )
    return 0.5 * S_textual + 0.5 * S_structural


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
    G: np.ndarray, S_textual: np.ndarray, S_structural: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """Find the best possible alignment matrix by brute-force
    searching the best possible threshold.

    :param G: gold alignment matrix
    :param S_textual: textual similarity matrix
    :param S_structural: structural similarity matrix
    :return: ``(best threshold, best alpha, best f1, best alignment matrix)``
    """
    alphas = np.arange(0.0, 1.0, 0.01)
    ts = np.arange(0.0, 1.0, 0.01)
    f1s = np.zeros((alphas.shape[0], ts.shape[0]))

    # Compute the best combination of both matrices
    # S_combined = α × S_textual + (1 - α) × S_structural
    print("searching for α and t...", file=sys.stderr)
    for alpha_i, alpha in tqdm(enumerate(alphas), total=alphas.shape[0]):
        S = alpha * S_textual + (1 - alpha) * S_structural
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
    best_S = best_alpha * S_textual + (1 - best_alpha) * S_structural
    best_M = best_S > best_t

    return (best_t, best_alpha, best_f1, best_M)


def threshold_align_blocks(
    S: np.ndarray, t: float, block_to_episode: np.ndarray
) -> np.ndarray:
    M_align_blocks = S >= t

    _, uniq_start_i = np.unique(block_to_episode, return_index=True)
    splits = np.split(M_align_blocks, uniq_start_i[1:], axis=0)

    M = []
    for split in splits:
        M.append(np.any(split, axis=0))

    M = np.stack(M)

    return M


def tune_threshold(
    X_tune: List[np.ndarray],
    G_tune: List[np.ndarray],
    threshold_search_space: np.ndarray,
    silent: bool = False,
) -> float:
    best_t = threshold_search_space[0]
    best_f1 = 0

    progress = tqdm(threshold_search_space, disable=silent)

    for t in progress:
        f1_list = []

        for X, G in zip(X_tune, G_tune):
            M = X > t
            f1 = precision_recall_fscore_support(
                G.flatten(), M.flatten(), average="binary", zero_division=0.0
            )[2]
            f1_list.append(f1)

        mean_f1 = sum(f1_list) / len(f1_list)
        progress.set_description(f"tuning ({t:.2f}, {mean_f1:.2f})")
        if mean_f1 > best_f1:
            best_t = t
            best_f1 = mean_f1

    return best_t


def tune_threshold_other_medias(
    media_pair: Literal["tvshow-novels", "comics-novels", "tvshow-comics"],
    sim_mode: Literal["structural", "textual", "combined"],
    threshold_search_space: np.ndarray,
    textual_sim_fn: Literal["tfidf", "sbert"] = "tfidf",
    structural_mode: Literal["edges", "nodes"] = "edges",
    structural_use_weights: bool = True,
    structural_filtering: Literal[
        "none", "common", "named", "common+named"
    ] = "common+named",
    silent: bool = False,
) -> float:
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

    return tune_threshold(X_tune, G_tune, threshold_search_space, silent=silent)
