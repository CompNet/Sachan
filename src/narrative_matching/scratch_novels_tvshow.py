# -*- eval: (code-cells-mode); -*-

# %%
import enum
from typing import *
import os, sys, glob, copy, pickle
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
from more_itertools import flatten


__file__ = os.path.expanduser("~/Dev/Sachan/src/matching/scratch.py")
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/.."
sys.path.append(f"{root_dir}/src")


def filtered_graph(G: nx.Graph, nodes: list) -> nx.Graph:
    H = copy.deepcopy(G)
    for node in nodes:
        if node in H:
            H.remove_node(node)
    return H


def keep_common_named_characters(
    novels_graphs: List[nx.Graph], comics_graphs: List[nx.Graph]
) -> Tuple[List[nx.Graph], List[nx.Graph]]:
    comics_characters = set(flatten([G.nodes for G in comics_graphs]))
    book_characters = set(flatten([G.nodes for G in novels_graphs]))
    comics_xor_book_characters = comics_characters ^ book_characters

    unnamed_characters = set()
    for G in comics_graphs + novels_graphs:
        for character in comics_characters | book_characters:
            if not G.nodes.get(character, {"named": True})["named"]:
                unnamed_characters.add(character)

    banned_characters = comics_xor_book_characters & unnamed_characters

    return (
        [filtered_graph(G, list(banned_characters)) for G in novels_graphs],
        [filtered_graph(G, list(banned_characters)) for G in comics_graphs],
    )


def jaccard_graph_sim(
    G: nx.Graph,
    H: nx.Graph,
    mode: Literal["nodes", "edges"],
    weights: Optional[dict] = None,
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
    comics_graphs: List[nx.Graph], novels_graphs: List[nx.Graph], kernel: Callable
) -> np.ndarray:
    """
    :return: ``(novels_chapters_nb, tvshow_units_nb)``
    """
    M = np.zeros((len(novels_graphs), len(comics_graphs)))

    for chapter_i, chapter_G in enumerate(tqdm(novels_graphs)):
        for scene_i, scene_G in enumerate(comics_graphs):
            M[chapter_i][scene_i] = kernel(chapter_G, scene_G)

    return M


def get_episode_i(G: nx.Graph) -> int:
    assert G.graph["season"] < 7
    return (G.graph["season"] - 1) * 10 + G.graph["episode"] - 1


def get_align_msim(
    M_sim: np.ndarray, M_block_to_episode: np.ndarray, threshold: float
) -> np.ndarray:
    """Given similarity between blocks and chapters, return the similarity between chapters and episodes.
    :param M_sim: ``(chapters_nb, blocks_nb)``
    :param M_block_to_episode: ``(blocks_nb)``
    :param threshold: between 0 and 1
    :return: M_split_sims ``(episodes_nb, chapters_nb)``
    """
    _, uniq_start_i = np.unique(M_block_to_episode, return_index=True)
    splits = np.split(M_sim, uniq_start_i[1:], axis=1)

    M_split_sims = []
    for split in splits:
        M_split_sims.append(np.max(split, axis=1))
    return np.stack(M_split_sims)


# %% Data loading
from sklearn.metrics import precision_recall_fscore_support


gold_alignment_path = "./tvshow_novels_gold_alignment.pickle"

novels_graphs = []
for path in sorted(glob.glob(f"{root_dir}/in/novels/instant/*.graphml")):
    novels_graphs.append(nx.read_graphml(path))
novels_graphs = [
    nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
    for G in novels_graphs
]
assert len(novels_graphs) > 0

tvshow_graphs = []
for path in sorted(
    glob.glob(f"{root_dir}/in/tvshow/instant/block_locations/*.graphml")
):
    tvshow_graphs.append(nx.read_graphml(path))
tvshow_graphs = [G for G in tvshow_graphs if G.graph["season"] < 7]
tvshow_graphs = [
    nx.relabel_nodes(G, {node: data["name"] for node, data in G.nodes(data=True)})
    for G in tvshow_graphs
]
assert len(tvshow_graphs) > 0

# * Gold alignment
with open(gold_alignment_path, "rb") as f:
    # (novels_chapters_nb, comics_chapters_nb)
    M_align_gold = pickle.load(f)

novels_graphs, tvshow_graphs = keep_common_named_characters(
    novels_graphs, tvshow_graphs
)

# %% Alignment computation
import os
from functools import partial

# * Jaccard index setup
tvshow_characters = set(flatten([G.nodes for G in tvshow_graphs]))
book_characters = set(flatten([G.nodes for G in novels_graphs]))
selected_characters = tvshow_characters | book_characters

# ** Nodes mode
characters_appearances = {char: 0 for char in selected_characters}
for G in tvshow_graphs + novels_graphs:
    for node in G.nodes:
        characters_appearances[node] += 1

# ** Edges mode
rel_appearances = {}
for G in tvshow_graphs + novels_graphs:
    for edge in G.edges:
        rel_appearances[tuple(sorted(edge))] = (
            rel_appearances.get(tuple(sorted(edge)), 0) + 1
        )

# ** Compute n^2 similarity
node_weights = {c: 1 / n for c, n in characters_appearances.items()}
edge_weights = {c: 1 / n for c, n in rel_appearances.items()}

# * Alignment
kernels: Dict[str, Dict[str, Any]] = {
    "jaccard index (nodes)": {"fn": partial(jaccard_graph_sim, mode="nodes")},
    "jaccard index (nodes, weighted)": {
        "fn": partial(jaccard_graph_sim, mode="nodes", weights=node_weights)
    },
    "jaccard index (edges)": {"fn": partial(jaccard_graph_sim, mode="edges")},
    "jaccard index (edges, weighted)": {
        "fn": partial(jaccard_graph_sim, mode="edges", weights=edge_weights)
    },
}

M_block_to_episode = np.array([get_episode_i(G) for G in tvshow_graphs])

for kernel_name, kernel_dict in tqdm(kernels.items()):

    # (novels_chapters_nb, tvshow_blocks_nb)
    M_sim = graph_similarity_matrix(tvshow_graphs, novels_graphs, kernel_dict["fn"])
    kernel_dict["M_sim_blocks"] = M_sim

    # Compute (precision, recall, F1) and the best threshold
    thresholds = np.arange(0, 1, 0.01)
    best_f1 = 0.0
    best_threshold = 0.0
    best_M_align = M_sim >= 0.0
    for threshold in thresholds:
        M_sim_episodes = get_align_msim(M_sim, M_block_to_episode, threshold)
        kernel_dict["M_sim_episodes"] = M_sim_episodes
        M_align = M_sim_episodes > threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            M_align_gold.flatten(),
            M_align.flatten(),
            average="binary",
            zero_division=0.0,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_M_align = M_align
            best_threshold = threshold

    kernel_dict["M_align"] = best_M_align
    kernel_dict["f1"] = best_f1


plt.figure(figsize=(8, 5))
plt.bar(kernels.keys(), [kdict["f1"] for kdict in kernels.values()])
plt.ylabel("F1 score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# %%
from nltk.tokenize import sent_tokenize
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open(os.path.expanduser("~/Nextcloud/phd/irp/chapters_summaries_short.txt")) as f:
    chapters_summaries = f.read().split("\n\n")[:-1]

with open(os.path.expanduser("~/Nextcloud/phd/irp/episodes_summaries.txt")) as f:
    episodes_summaries = f.read().split("\n\n")

episodes_summaries = episodes_summaries[:60]

vectorizer = TfidfVectorizer()
v = vectorizer.fit(chapters_summaries + episodes_summaries)
v = vectorizer.fit_transform(chapters_summaries + episodes_summaries)
chapters_v, episodes_v = (v[: len(chapters_summaries)], v[len(chapters_summaries) :])

M_sim_tfidf = cosine_similarity(chapters_v, episodes_v).T

plt.imshow(M_sim_tfidf > 0.2)
plt.show()

# %%
from nltk.tokenize import sent_tokenize
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open(os.path.expanduser("~/Nextcloud/phd/irp/chapters_summaries_short.txt")) as f:
    chapters_summaries = f.read().split("\n\n")[:-1]

with open(os.path.expanduser("~/Nextcloud/phd/irp/episodes_summaries.txt")) as f:
    episodes_summaries = f.read().split("\n\n")

episodes_summaries = episodes_summaries[:60]

vectorizer = TfidfVectorizer()
v = vectorizer.fit(chapters_summaries + episodes_summaries)

chapters_v = vectorizer.transform(chapters_summaries)

M_sim_tfidf = np.zeros((len(episodes_summaries), len(chapters_summaries)))
for e_i, e_sum in enumerate(episodes_summaries):
    e_sum_v = vectorizer.transform(sent_tokenize(e_sum))
    sims = np.max(cosine_similarity(e_sum_v, chapters_v), axis=0)
    M_sim_tfidf[e_i] = sims

plt.imshow(M_sim_tfidf)
plt.show()

# %% SentenceBert version
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt

stransformer = SentenceTransformer("all-mpnet-base-v2")

with open(os.path.expanduser("~/Nextcloud/phd/irp/chapters_summaries_short.txt")) as f:
    chapters_summaries = f.read().split("\n\n")[:-1]

with open(os.path.expanduser("~/Nextcloud/phd/irp/episodes_summaries.txt")) as f:
    episodes_summaries = f.read().split("\n\n")

episodes_summaries = episodes_summaries[:60]

chapters_v = stransformer.encode(chapters_summaries)

M_sim_tfidf = np.zeros((len(episodes_summaries), len(chapters_summaries)))
for e_i, e_sum in enumerate(tqdm(episodes_summaries)):
    e_sum_v = stransformer.encode(sent_tokenize(e_sum))
    sims = np.max(cosine_similarity(e_sum_v, chapters_v), axis=0)
    M_sim_tfidf[e_i] = sims

plt.imshow(M_sim_tfidf > 0.65)
plt.show()

# %% DEV
M_sim_graphs = kernels["jaccard index (edges, weighted)"]["M_sim_episodes"]

fig, axs = plt.subplots(1, 2)
axs[0].imshow(M_sim_graphs)
axs[1].imshow(M_sim_tfidf)
plt.show()

#%% Best TF-IDF match
ep_limit = 10
ch_limit = 73

best_tfidf_f1 = 0.0
best_tfidf_t = 0.0
for t in np.arange(0.0, 1.0, 0.01):
    M_align = M_sim_tfidf > t
    _, _, tfidf_f1, _ = precision_recall_fscore_support(
        M_align_gold[:ep_limit, :ch_limit].flatten(),
        M_align[:ep_limit, :ch_limit].flatten(),
        average="binary",
        zero_division=0.0,
    )
    if tfidf_f1 > best_tfidf_f1:
        best_tfidf_f1 = tfidf_f1
        best_tfidf_t = t

print(f"{best_tfidf_f1=}")
print(f"{best_tfidf_t=}")

# %% Best graph matching
ep_limit = 10
ch_limit = 73

best_graph_f1 = 0.0
best_graph_t = 0.0
for t in np.arange(0.0, 1.0, 0.01):
    M_align = M_sim_graphs > t
    _, _, graph_f1, _ = precision_recall_fscore_support(
        M_align_gold[:ep_limit, :ch_limit].flatten(),
        M_align[:ep_limit, :ch_limit].flatten(),
        average="binary",
        zero_division=0.0,
    )
    if graph_f1 > best_graph_f1:
        best_graph_f1 = graph_f1
        best_graph_t = t

print(f"{best_graph_f1=}")
print(f"{best_graph_t=}")


# %% Best combination
from tqdm import tqdm

ep_limit = 10
ch_limit = 73

alphas = np.arange(0.0, 1.0, 0.01)
ts = np.arange(0.0, 1.0, 0.01)
f1s = np.zeros((alphas.shape[0], ts.shape[0]))

for alpha_i, alpha in tqdm(enumerate(alphas), total=alphas.shape[0]):
    M_sim = alpha * M_sim_tfidf + (1 - alpha) * M_sim_graphs
    for t_i, t in enumerate(ts):
        M_align = M_sim > t
        precision, recall, f1, _ = precision_recall_fscore_support(
            M_align_gold[:ep_limit, :ch_limit].flatten(),
            M_align[:ep_limit, :ch_limit].flatten(),
            average="binary",
            zero_division=0.0,
        )
        f1s[alpha_i][t_i] = f1

best_f1_loc = np.argwhere(f1s == np.max(f1s))[0]
best_alpha = best_f1_loc[0] / 100.0
best_t = best_f1_loc[1] / 100.0
best_M_sim = best_alpha * M_sim_tfidf + (1 - best_alpha) * M_sim_graphs
print(f"{best_alpha=}")
print(f"{best_t=}")
print(f"best F1: {np.max(f1s)}")


fig, axs = plt.subplots(3, 1)
axs[0].imshow(f1s)
axs[0].set_xlabel("t")
axs[0].set_ylabel("alpha")
axs[1].imshow(best_M_sim[:ep_limit, :ch_limit] > best_t)
axs[2].imshow(M_align_gold[:ep_limit, :ch_limit])
plt.show()
