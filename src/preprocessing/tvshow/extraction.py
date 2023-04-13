# Several functions used to extract TVShow networks
#
# Author: Arthur Amalvy
# 04/2023
from typing import Dict, List, Literal
from collections import defaultdict
import json, os, itertools
from tqdm import tqdm
import numpy as np
import networkx as nx
import pandas as pd


def load_tvshow_character_map(path: str) -> Dict[str, str]:
    """Load the character canonical names map from the ``charmap.csv`` file."""
    df = pd.read_csv(os.path.expanduser(path))
    charmap = {}
    for _, row in df.iterrows():
        tvshow_name = row["TvShowName"]
        # account for NaN: some characters do not have a normalized
        # name since they do not have a proper name (for example,
        # 'Bolton Soldier'). In that case, we simply use the TVShow
        # name.
        canonical_name = (
            row["NormalizedName"]
            if isinstance(row["NormalizedName"], str)
            else tvshow_name
        )
        charmap[tvshow_name] = canonical_name
    return charmap


def load_got_tvshow_graphs(
    path: str, granularity: Literal["episode", "scene"], charmap: Dict[str, str]
) -> List[nx.Graph]:
    """Load character networks from the TVShow, using Jeffrey
    Lancaster's Github repository.

    :param path: path to Jeffrey Lancaster's game-of-thrones
        repository
    :param granularity:
    :param charmap: A map from each character name in the TVShow to
        its canonical name.  If a character is not in this map, it
        will be _ignored_ by this function and wont appear in the
        final graph.

    :return: a ``nx.Graph`` for each scene
    """
    root_dir = os.path.expanduser(path)

    # * Load main 'episodes.json' file
    episodes_path = os.path.join(root_dir, "data", "episodes.json")
    with open(episodes_path) as f:
        got_data = json.load(f)

    # * Load characters info file
    characters_path = os.path.join(root_dir, "data", "characters.json")
    with open(characters_path) as f:
        characters_data = json.load(f)

    # * Load characters sex info file
    sex_path = os.path.join(root_dir, "data", "characters-gender-all.json")
    with open(sex_path) as f:
        sex_data = json.load(f)
    male_characters = set(sex_data["male"])
    female_characters = set(sex_data["female"])

    # * Utils
    def get_sex(character: str) -> Literal["Male", "Female", "Unknown"]:
        """
        :param character: name of the character in
            ``'characters-gender-all.json'``
        """
        if character in male_characters:
            return "Male"
        elif character in female_characters:
            return "Female"
        else:
            return "Unknown"

    def get_houses(character: dict) -> str:
        houses = character.get("houseName")
        if houses is None:
            return ""
        elif isinstance(houses, list):
            return " ".join(houses)
        else:
            return houses

    # * Load characters attributes
    # { canonical name => { attribute_name => attribute_value } }
    characters_attributes = {}
    for character_data in characters_data["characters"]:
        canonical_name = charmap.get(character_data["characterName"])
        if canonical_name is None:
            continue
        characters_attributes[canonical_name] = {
            "house": get_houses(character_data),
            "sex": get_sex(character_data["characterName"]),
        }

    # * Parsing of episodes.json
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
                canonical_name = charmap.get(character["name"])
                # Character with no canonical names are ignored
                if canonical_name is None:
                    continue
                attributes = characters_attributes.get(
                    canonical_name, {"house": "", "sex": "Unknown"}
                )
                G.add_node(canonical_name, **attributes)

            for c1, c2 in itertools.combinations(scene["characters"], 2):
                n1 = charmap.get(c1["name"])
                n2 = charmap.get(c2["name"])
                # Character with no canonical names are ignored
                if n1 is None or n2 is None:
                    continue
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1
                else:
                    G.add_edge(n1, n2, weight=1)

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
