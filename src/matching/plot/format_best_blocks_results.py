import argparse, os, pickle
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--format", type=str, default="latex", help="either 'plain' or 'latex'"
    )
    args = parser.parse_args()

    dfs_dict = {}

    for medias in ["tvshow-novels", "comics-novels"]:
        with open(
            f"{root_dir}/out/matching/plot/{medias}_structural_blocks/df.pickle", "rb"
        ) as f:
            df = pickle.load(f)
            df = df.loc[
                :, ["sim_mode", "use_weights", "character_filtering", "alignment", "f1"]
            ]

            align_map = {
                "smith-waterman": "Smith--Waterman",
                "threshold": "Thresholding",
            }
            df["alignment"] = df["alignment"].apply(align_map.get)

            c_filtering_remap = {
                "named": "named",
                "common": "common",
                "top20s5": "top-20",
                "top20s2": "top-20",
            }
            df["character_filtering"] = df["character_filtering"].apply(
                c_filtering_remap.get
            )

            dfs_dict[medias] = df

    indexs = ["sim_mode", "use_weights", "character_filtering", "alignment"]
    # output cols: f1_x, f1_y
    df = dfs_dict["tvshow-novels"].merge(
        dfs_dict["comics-novels"], how="inner", on=indexs
    )
    df = df.rename(
        columns={
            "f1_x": "Novels vs. TV Show",
            "f1_y": "Novels vs. Comics",
            "alignment": "Alignment",
        }
    )

    df = df.set_index(["sim_mode", "use_weights", "character_filtering", "Alignment"])
    # fix the order of pairs
    df = df[["Novels vs. Comics", "Novels vs. TV Show"]]
    # put the "character filtering" level in the column
    df = df.unstack(2)
    # fix the order of the character sets
    df = df.reindex(["named", "common", "top-20"], axis=1, level=1)
    df = df.groupby("Alignment").max()

    # sort alignment so that thresholding comes first
    df = df.sort_values(by="Alignment", ascending=False)

    if args.format == "plain":
        print(df)
        exit(0)

    def format_index(value: str):
        # character filtering sets: monospace style
        if value in ["common", "named", "top-20"]:
            return "ttfamily: ;"
        # media pairs: italic style
        else:
            return "itshape: ;"

    LaTeX_export = (
        df.style.format(lambda v: "{:.2f}".format(v * 100))
        .map_index(format_index, axis="columns")
        .highlight_max(subset="Novels vs. Comics", props="bfseries: ;", axis=None)
        .highlight_max(subset="Novels vs. TV Show", props="bfseries: ;", axis=None)
        .to_latex(hrules=True, column_format="l r@{~}r@{~}r r@{~}r@{~}r")
    )
    # hide the "character_filtering" level name
    LaTeX_export = LaTeX_export.replace("character_filtering", "")

    # set the level titles to be at the same place as the media pairs
    # columns
    lines = LaTeX_export.split("\n")
    LEVELS_TITLES_I = 4
    TITLES_NB = 1
    levels_titles = lines[LEVELS_TITLES_I]
    titles = levels_titles.split("&")[:TITLES_NB]
    lines[2] = "&".join(titles) + "& " + lines[2][lines[2].index("\\") :]
    lines = lines[:LEVELS_TITLES_I] + lines[LEVELS_TITLES_I + 1 :]
    LaTeX_export = "\n".join(lines)

    print(LaTeX_export)
