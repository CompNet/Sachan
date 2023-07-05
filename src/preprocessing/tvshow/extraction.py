# Several functions used to extract TVShow networks
#
# Author: Arthur Amalvy
# 04/2023
from typing import Any, Dict, List, Literal, Optional
import json, os, itertools
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


def load_characters_csv(path: str, **kwargs) -> pd.DataFrame:
    """Load the 'characters.csv' file as a pandas dataframe.

    :param kwargs: kwargs passed to ``pd.read_csv``
    """
    if not "sep" in kwargs:
        kwargs["sep"] = ";"
    return pd.read_csv(path, **kwargs)


def _parse_episodes_json_episode_graphs(
    got_data: dict,
    charmap: Dict[str, str],
    char_attrs_map: Dict[str, Dict[str, Any]],
    default_attrs: Dict[str, Any],
) -> List[nx.Graph]:
    """Parse ``episodes.json`` into a graph per episode

    :param got_datas: ``episodes.json`` file as a dict
    :param charmap: A map from each character name in the TVShow to
        its canonical name.  If a character is not in this map, it
        will be _ignored_ by this function and wont appear in the
        final graph.
    :param char_attrs_map: ``{ canonical_name => { attr_name =>
        attr_value } }``
    :param default_attrs: default attributes for characters
    """
    graphs = []

    for episode in got_data["episodes"]:
        G = nx.Graph()
        G.graph["season"] = episode["seasonNum"]
        G.graph["episode"] = episode["episodeNum"]

        for scene in episode["scenes"]:
            for character in scene["characters"]:
                canonical_name = charmap.get(character["name"])
                # Character with no canonical names are ignored
                if canonical_name is None:
                    continue
                attributes = char_attrs_map.get(canonical_name, default_attrs)
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

        graphs.append(G)

    return graphs


def _parse_episodes_json_scene_graphs(
    got_data: dict,
    charmap: Dict[str, str],
    char_attrs_map: Dict[str, Dict[str, Any]],
    default_attrs: Dict[str, Any],
) -> List[nx.Graph]:
    """Parse ``episodes.json`` into a graph per scene

    :param got_datas: ``episodes.json`` file as a dict
    :param charmap: A map from each character name in the TVShow to
        its canonical name.  If a character is not in this map, it
        will be _ignored_ by this function and wont appear in the
        final graph.
    :param char_attrs_map: ``{ canonical_name => { attr_name =>
        attr_value } }``
    :param default_attrs: default attributes for characters
    """
    # * Parsing of episodes.json
    graphs = []

    for episode in got_data["episodes"]:
        season_i = episode["seasonNum"]
        episode_i = episode["episodeNum"]

        for scene_i, scene in enumerate(episode["scenes"]):
            G = nx.Graph()
            G.graph["season"] = season_i
            G.graph["episode"] = episode_i
            G.graph["scene"] = scene_i

            for character in scene["characters"]:
                canonical_name = charmap.get(character["name"])
                # Character with no canonical names are ignored
                if canonical_name is None:
                    continue
                attributes = char_attrs_map.get(canonical_name, default_attrs)
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

            graphs.append(G)

    return graphs


def _parse_episodes_json_block_graphs(
    got_data: dict,
    charmap: Dict[str, str],
    char_attrs_map: Dict[str, Dict[str, Any]],
    default_attrs: Dict[str, Any],
    method: Literal["locations", "similarity"],
    method_kwargs: Optional[dict],
) -> List[nx.Graph]:
    """Parse ``episodes.json`` into a graph per 'block' (several
    consecutive scenes).  The definition of a 'block' depends on
    ``method``.

    :param got_datas: ``episodes.json`` file as a dict

    :param charmap: A map from each character name in the TVShow to
        its canonical name.  If a character is not in this map, it
        will be _ignored_ by this function and wont appear in the
        final graph.

    :param char_attrs_map: ``{ canonical_name => { attr_name =>
        attr_value } }``

    :param default_attrs: default attributes for characters

    :param method: block groupment method :

            - ``'locations'``: cut blocks by considering that
              consecutive scenes with common locations are in the same
              block.

            - ``'similarity'``: cut blocks by considering that
              consecutive scenes with similar enough character graphs
              are in the same block.  Similarity is computed using
              Jaccard index on nodes.  If this method is chosen, the
              caller must supply a ``'threshold'`` kwargs in
              ``method_kwargs``.

    :param method_kwargs: additional kwargs for the chosen ``method``.
    """
    assert method in ["locations", "similarity"]

    # * Parsing of episodes.json
    graphs = []

    # * This function determines if a scene starts a new block, using
    # A the previous scene. Depends on the supplied method.
    def scene_starts_new_block(prev_scene: Optional[dict], scene: dict) -> bool:
        if prev_scene is None:
            return False

        if method == "locations":
            prev_locations = [
                l
                for l in [
                    prev_scene.get("location"),
                    prev_scene.get("subLocation"),
                ]
                if not l is None
            ]
            return (
                not scene.get("location") in prev_locations
                and not scene.get("subLocation") in prev_locations
            )

        elif method == "similarity":
            characters = {c["name"] for c in scene["characters"]}
            prev_characters = {c["name"] for c in prev_scene["characters"]}
            intersection = characters.intersection(prev_characters)
            union = characters.union(prev_characters)
            if len(union) == 0:
                return True
            similarity = len(intersection) / len(union)
            assert not method_kwargs is None
            return similarity < method_kwargs["threshold"]

    for episode in got_data["episodes"]:
        G = nx.Graph()
        G.graph["season"] = episode["seasonNum"]
        G.graph["episode"] = episode["episodeNum"]

        prev_scene = None

        for scene in episode["scenes"]:
            if scene_starts_new_block(prev_scene, scene):
                graphs.append(G)  # flush current block
                G = nx.Graph()
                G.graph["season"] = episode["seasonNum"]
                G.graph["episode"] = episode["episodeNum"]

            for character in scene["characters"]:
                canonical_name = charmap.get(character["name"])
                # Character with no canonical names are ignored
                if canonical_name is None:
                    continue
                attributes = char_attrs_map.get(canonical_name, default_attrs)
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

            prev_scene = scene

        # flush last episode graph
        graphs.append(G)

    return graphs


def load_got_tvshow_graphs(
    path: str,
    granularity: Literal["episode", "scene", "block"],
    charmap: Dict[str, str],
    block_method: Optional[Literal["locations", "similarity"]] = None,
    block_method_kwargs: Optional[dict] = None,
) -> List[nx.Graph]:
    """Load character networks from the TVShow, using Jeffrey
    Lancaster's Github repository.

    :param path: path to Jeffrey Lancaster's game-of-thrones
        repository

    :param granularity: either :

            - ``'episode'``: A graph per episode

            - ``'scene'``: A graph per scene (as defined is Jeffrey
              Lancaster data)

            - ``'block'``: A graph per 'block'.  A block is a group of
              scene corresponding to a PoV

    :param charmap: A map from each character name in the TVShow to
        its canonical name.  If a character is not in this map, it
        will be _ignored_ by this function and wont appear in the
        final graph.

    :param block_method: method to use for determining blocks - see
        :func:`_parse_episodes_json_block_graphs`

    :param block_method_kwargs: additional kwargs for the method use
        to determine blocks - see
        :func:`_parse_episodes_json_block_graphs`

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
    default_char_attrs = {"house": "", "sex": "Unknown"}
    if granularity == "episode":
        return _parse_episodes_json_episode_graphs(
            got_data, charmap, characters_attributes, default_char_attrs
        )
    elif granularity == "scene":
        return _parse_episodes_json_scene_graphs(
            got_data, charmap, characters_attributes, default_char_attrs
        )
    elif granularity == "block":
        assert not block_method is None
        if block_method_kwargs is None:
            block_method_kwargs = {}
        return _parse_episodes_json_block_graphs(
            got_data,
            charmap,
            characters_attributes,
            default_char_attrs,
            block_method,
            block_method_kwargs,
        )
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


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
