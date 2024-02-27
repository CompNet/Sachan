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
    for medias in ["tvshow-novels", "tvshow-comics", "comics-novels"]:
        with open(
            f"{root_dir}/out/matching/plot/{medias}_combined/df.pickle", "rb"
        ) as f:
            df = pickle.load(f)
            df = df.loc[
                :,
                [
                    "textual_sim_fn",
                    "structural_sim_mode",
                    "structural_use_weights",
                    "structural_character_filtering",
                    "alignment",
                    "f1",
                ],
            ]

            align_map = {
                "smith-waterman": "Smith--Waterman",
                "threshold": "Thresholding",
            }
            df["alignment"] = df["alignment"].apply(align_map.get)

            dfs_dict[medias] = df

    indexs = [
        "textual_sim_fn",
        "structural_sim_mode",
        "structural_use_weights",
        "structural_character_filtering",
        "alignment",
    ]
    # output cols: f1_x, f1_y
    df = dfs_dict["tvshow-novels"].merge(
        dfs_dict["tvshow-comics"], how="inner", on=indexs
    )
    # output cols: f1_x, f1_y, f1
    df = df.merge(dfs_dict["comics-novels"], how="inner", on=indexs)
    df = df.rename(
        columns={
            "f1_x": "Novels vs. TV Show",
            "f1_y": "Comics vs. TV Show",
            "f1": "Novels vs. Comics",
            "alignment": "Alignment",
        }
    )
    df = df.set_index(indexs[:-1] + ["Alignment"])
    # fix the order of pairs
    df = df[["Novels vs. Comics", "Novels vs. TV Show", "Comics vs. TV Show"]]

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
        .highlight_max(props="bfseries: ;", axis=0)
        .to_latex(
            hrules=True,
            sparse_index=True,
            multicol_align="c",
            multirow_align="c",
            column_format="lrrr",
        )
    )
    # HACK: things we can't do with df.style.format.to_latex...
    # hide the "character_filtering" level name
    LaTeX_export = LaTeX_export.replace("character_filtering", "")
    # this one is the trickiest: we set the level titles to be at
    # the same place as the media pairs columns
    # basically this head:
    #
    #                                                          Novels vs. Comics  ...
    #                                                            named    common  ...
    # Representation Measure         Alignment                                    ...
    # Edges          Jaccard         Smith--Waterman          0.545455  0.538462  ...
    #                                Thresholding             0.258621  0.284615  ...
    #                Ru\v{z}i\v{c}ka Smith--Waterman          0.556338  0.575439  ...
    #                                Thresholding             0.296053  0.347439  ...
    #
    # must be transformed into this head:
    #
    # Representation Measure         Alignment                 Novels vs. Comics  ...
    #                                                            named    common  ...
    # Edges          Jaccard         Smith--Waterman          0.545455  0.538462  ...
    #                                Thresholding             0.258621  0.284615  ...
    #                Ru\v{z}i\v{c}ka Smith--Waterman          0.556338  0.575439  ...
    #
    # ... and it's the only way I found! Sorry!
    lines = LaTeX_export.split("\n")
    LEVELS_TITLES_I = 3
    TITLES_NB = 1
    levels_titles = lines[LEVELS_TITLES_I]
    titles = levels_titles.split("&")[:TITLES_NB]
    lines[2] = "&".join(titles) + "& " + lines[2][lines[2].index("\\") :]
    lines = lines[:LEVELS_TITLES_I] + lines[LEVELS_TITLES_I + 1 :]
    LaTeX_export = "\n".join(lines)

    print(LaTeX_export)
