from typing import List, Literal
from collections import defaultdict
import re, json, math, os, itertools
from tqdm import tqdm
import numpy as np
import networkx as nx
import pandas as pd


def _is_chapter_row(row: pd.core.series.Series) -> bool:
    return (
        isinstance(row["Character"], str)
        and not re.search(r"CHAPTER|PROLOGUE|EPILOGUE", row["Character"]) is None
        and all(
            [
                isinstance(row[key], float) and math.isnan(row[key])
                for key in row.keys()
                if not key == "Character"
            ]
        )
    )


def _row_links(row: pd.core.series.Series) -> List[str]:
    links_keys = [k for k in row.keys() if k.startswith("Friendly Link")] + [
        k for k in row.keys() if k.startswith("Hostile Link")
    ]
    return [row[key].strip() for key in links_keys if isinstance(row[key], str)]


def load_got_book_graphs(path: str) -> List[nx.Graph]:
    """
    :param path: path to the book CSV
    :return: a :class:`nx.Graph` per chapter
    """
    df = pd.read_csv(path)
    graphs = []
    G = nx.Graph()
    for i, row in tqdm(df.iterrows(), total=len(df)):  # type: ignore
        if _is_chapter_row(row):
            if len(G) > 0:
                graphs.append(G)
            G = nx.Graph()
            continue
        character = row["Character"].strip()
        G.add_node(character)
        links = _row_links(row)
        for link in links:
            if not (character, link) in G.edges:
                G.add_edge(character, link, weight=0)
            G[character][link]["weight"] += 1
    graphs.append(G)

    return graphs


def load_got_book_chapters(path: str) -> List[str]:
    """
    :param path: path to the GOT's .txt
    :return: a :class:`str` per chapter
    """
    with open(path) as f:
        text = f.read()
    return re.split(r"\n\n\n[A-Z]+\n\n\n", text)


def load_got_tvshow_conversational_scene_graphs(path: str) -> List[nx.Graph]:
    """
    :param path: path to the graphml file obtained using
        https://github.com/bostxavier/Narrative-Smoothing
    :return: a :class:`nx.Graph` per scene
    """

    G = nx.read_graphml(os.path.expanduser(path))

    scenes = [
        (u, v, scene_id, weight)
        for u, v, scene_id, weight in G.edges(data="weight", keys=True)
    ]
    scenes = sorted(scenes, key=lambda s: s[2])

    # global graph to keep track of all relationships
    H = nx.Graph()
    for u, v, _, _ in scenes:
        H.add_edge(u, v, weight=0.0)

    # { scene_id => graph }
    graphs = defaultdict(nx.Graph)

    for u, v, scene_id, weight in tqdm(scenes):  # type: ignore
        G = graphs[scene_id]
        if weight > H[u][v]["weight"]:
            if not (u, v) in G:
                G.add_edge(u, v, weight=0)
            G[u][v]["weight"] += 1
            H[u][v]["weight"] = weight

    graphs = [(G, scene_id) for scene_id, G in graphs.items()]
    graphs = sorted(graphs, key=lambda gi: gi[1])
    return [G for G, _ in graphs]


def _scene_speech_segments(
    scene: dict, scenes: List[dict], speech_segments: List[dict]
) -> List[dict]:
    scene_starts = [s["start"] for s in scenes]

    possible_scene_ends = [i for i in scene_starts if i > scene["start"]]
    if len(possible_scene_ends) == 0:
        scene_end = float("inf")
    else:
        scene_end = min(possible_scene_ends)

    return [
        ss
        for ss in speech_segments
        if ss["start"] >= scene["start"] and ss["end"] <= scene_end
    ]


def load_got_tvshow_conversational_episode_graphs(
    path: str, season: Literal[1, 2, 3, 4, 5, 6, 7, 8]
) -> List[nx.Graph]:
    """
    :param path: path to the ``'got.json'`` file
    """
    with open(os.path.expanduser(path)) as f:
        got_datas = json.load(f)

    episodes = []

    for episode in got_datas["seasons"][season - 1]["episodes"]:

        G = nx.Graph()

        episode_scenes = episode["data"]["scenes"]
        episode_speech_segments = episode["data"]["speech_segments"]

        for scene in episode_scenes:

            scene_speech_segments = _scene_speech_segments(
                scene, episode_scenes, episode_speech_segments
            )

            scene_speakers = [s["speaker"] for s in scene_speech_segments]
            for speaker in scene_speakers:
                if not speaker in G:
                    G.add_node(speaker)
            for s1, s2 in itertools.combinations(scene_speakers, 2):
                if s1 == s2:
                    continue
                G.add_edge(s1, s2, weight=0)
                G[s1][s2]["weight"] += 1

        episodes.append(G)

    return episodes


def load_got_tvshow_scene_dialogues(
    path: str, season: Literal[1, 2, 3, 4, 5, 6, 7, 8], strip_empty_scenes: bool = True
) -> List[str]:
    """
    :param path: path to the decrypted ``'got.json'`` file (obtained
        with https://github.com/bostxavier/Serial-Speakers)
    """
    with open(os.path.expanduser(path)) as f:
        got_datas = json.load(f)

    scenes = []

    for episode in got_datas["seasons"][season - 1]["episodes"]:

        episode_scenes = [[] for _ in episode["data"]["scenes"]]
        scene_starts = [scene["start"] for scene in episode["data"]["scenes"]]

        for speech_segment in episode["data"]["speech_segments"]:

            # determine the scene of the speech segment
            speech_segment_scene = None
            for i, start in enumerate(scene_starts):
                if speech_segment["start"] < start:
                    assert i > 0
                    speech_segment_scene = i - 1
                    break
            if speech_segment_scene is None:
                speech_segment_scene = len(scene_starts) - 1

            speaker = speech_segment["speaker"]
            text = speech_segment["text"]
            episode_scenes[speech_segment_scene].append(f"{speaker}: {text}")

        scenes += ["\n".join(scene) for scene in episode_scenes]

    if strip_empty_scenes:
        scenes = [s for s in scenes if not s == ""]

    return scenes


def load_got_tvshow_episodes(
    path: str, season: Literal[1, 2, 3, 4, 5, 6, 7, 8]
) -> List[str]:
    """
    :param path: path to the decrypted ``'got.json'`` file (obtained
        with https://github.com/bostxavier/Serial-Speakers)
    :return: dialogue for each episode
    """
    with open(os.path.expanduser(path)) as f:
        got_datas = json.load(f)

    episodes = []

    for episode in got_datas["seasons"][season - 1]["episodes"]:

        episode_text = []

        for speech_segment in episode["data"]["speech_segments"]:
            speaker = speech_segment["speaker"]
            text = speech_segment["text"]
            episode_text.append(f"{speaker}: {text}")

        episodes.append("\n".join(episode_text))

    return episodes


def load_got_tvshow_graphs(
    path: str, granularity: Literal["episode", "scene"]
) -> List[nx.Graph]:
    """
    :param path: path to Jeffrey Lancaster's game-of-thrones repository
    :return: a ``nx.Graph`` for each scene
    """
    root_dir = os.path.expanduser(path)
    episodes_path = os.path.join(root_dir, "data", "episodes.json")
    with open(episodes_path) as f:
        got_data = json.load(f)

    characters_path = os.path.join(root_dir, "data", "characters.json")
    with open(characters_path) as f:
        characters_data = json.load(f)

    gender_path = os.path.join(root_dir, "data", "characters-gender-all.json")
    with open(gender_path) as f:
        genders_data = json.load(f)
    male_characters = set(genders_data["male"])
    female_characters = set(genders_data["female"])

    def get_gender(character: str) -> Literal["male", "female", "unknown"]:
        if character in male_characters:
            return "male"
        elif character in female_characters:
            return "female"
        else:
            return "unknown"

    def get_houses(character: dict) -> str:
        houses = character.get("houseName")
        if houses is None:
            return ""
        elif isinstance(houses, list):
            return " ".join(houses)
        else:
            return houses

    characters_attributes = {
        character["characterName"]: {
            "house": get_houses(character),
            "gender": get_gender(character["characterName"]),
        }
        for character in characters_data["characters"]
    }

    graphs = []

    for episode in got_data["episodes"]:

        season_i = episode["seasonNum"]
        episode_i = episode["episodeNum"]

        G = None
        if granularity == "episode":
            G = nx.Graph()
            G.graph["season"] = season_i
            G.graph["episode"] = episode_i

        for scene_i, scene in enumerate(episode["scenes"]):

            if granularity == "scene":
                G = nx.Graph()
                G.graph["season"] = season_i
                G.graph["episode"] = episode_i
                G.graph["scene"] = scene_i
            assert not G is None

            for character in scene["characters"]:
                name = character["name"]
                attributes = characters_attributes.get(
                    name, {"house": "", "gender": "unknown"}
                )
                G.add_node(name, **attributes)

            for c1, c2 in itertools.combinations(scene["characters"], 2):
                if c1["name"] == c2["name"]:
                    continue
                G.add_edge(c1["name"], c2["name"], weight=1)

            if granularity == "scene":
                graphs.append(G)

        if granularity == "episode":
            graphs.append(G)

    return graphs


def load_got_episodes_chapters_alignment_matrix(path: str) -> np.ndarray:
    """
    :param path: path to the got-book-show repository
        (https://github.com/Joeltronics/got-book-show)
    :return: ``(episodes_nb, chapters_nb)``
    """
    root_dir = os.path.expanduser(path)

    BOOK_NAMES = [
        "A Game of Thrones",
        "A Clash of Kings",
        "A Storm of Swords",
        "A Feast for Crows",
        "A Dance with Dragons",
        "The Winds of Winter",
    ]
    BOOK_CHAPTERS_NB = [73, 70, 82, 46, 73, 26]
    SEASON_EPISODES_NB = 10
    SEASONS_NB = 6

    connections_path = os.path.join(root_dir, "input", "connections.csv")
    connections_df = pd.read_csv(connections_path).dropna(how="all")
    connections_df["Season"] = connections_df["Season"].astype(int)
    connections_df["Episode"] = connections_df["Episode"].astype(int)
    connections_df["Book"] = connections_df["Book"].astype(int)

    chapters_path = os.path.join(root_dir, "input", "chapters.csv")
    chapters_df = (
        pd.read_csv(chapters_path)
        .dropna(how="all")
        .rename({"Unnamed: 0": "Book"}, axis=1)
    )
    chapters_df["Chapter number in book"] = chapters_df[
        "Chapter number in book"
    ].astype(int)

    # (episodes_nb, chapters_nb)
    M = np.zeros((SEASONS_NB * SEASON_EPISODES_NB, len(chapters_df)))

    for _, row in connections_df.iterrows():
        # episode index
        episode_i = (row["Season"] - 1) * SEASON_EPISODES_NB + row["Episode"] - 1
        # chapter index
        book_i = row["Book"] - 1
        book_name = BOOK_NAMES[book_i]
        chapter_name = row["Chapter"]
        chapter_line = chapters_df.loc[
            (chapters_df["Chapter name"] == chapter_name)
            & (chapters_df["Book"] == book_name)
        ].iloc[0]
        chapter_i = (
            sum([n for i, n in enumerate(BOOK_CHAPTERS_NB) if i < book_i])
            + chapter_line["Chapter number in book"]
        )
        # update matrix with episode-chapter correspondance
        M[episode_i][chapter_i] = 1

    return M
