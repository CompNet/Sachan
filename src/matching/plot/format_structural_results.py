import argparse, os, pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--format", type=str, default="latex", help="either 'plain' or 'latex'"
    )
    parser.add_argument(
        "-e",
        "--best",
        action="store_true",
        help="if specified, simplify table and take the best score along jaccard and weighting parameters.",
    )
    parser.add_argument(
        "-c",
        "--cumulative",
        action="store_true",
        help="if specified, use cumulative results",
    )
    parser.add_argument(
        "-b", "--blocks", action="store_true"
    )
    args = parser.parse_args()

    dfs_dict = {}
    for medias in ["tvshow-novels", "tvshow-comics", "comics-novels"]:
        path = f"{root_dir}/out/matching/plot/{medias}_structural/df.pickle"
        if args.cumulative:
            path = (
                f"{root_dir}/out/matching/plot/{medias}_structural_cumulative/df.pickle"
            )
        if args.blocks:
            path = f"{root_dir}/out/matching/plot/{medias}_structural_blocks/df.pickle"
        with open(path, "rb") as f:
            df = pickle.load(f)
            df = df.loc[
                :, ["sim_mode", "use_weights", "character_filtering", "alignment", "f1"]
            ]

            c_filtering_remap = {
                "named": "named",
                "common": "common",
                "top20s5": "top-20",
                "top20s2": "top-20",
            }
            df["character_filtering"] = df["character_filtering"].apply(
                c_filtering_remap.get
            )

            weighted_remap = {True: "Ru\\v{z}i\\v{c}ka", False: "Jaccard"}
            df["use_weights"] = df["use_weights"].apply(weighted_remap.get)

            jaccard_remap = {"edges": "Edges", "nodes": "Vertices"}
            df["sim_mode"] = df["sim_mode"].apply(jaccard_remap.get)

            align_map = {
                "smith-waterman": "Smith--Waterman",
                "threshold": "Thresholding",
            }
            df["alignment"] = df["alignment"].apply(align_map.get)

            dfs_dict[medias] = df

    indexs = ["sim_mode", "use_weights", "character_filtering", "alignment"]
    # output cols: f1_x, f1_y
    df = dfs_dict["tvshow-novels"].merge(
        dfs_dict["tvshow-comics"], how="inner", on=indexs
    )
    # output cols: f1_x, f1_y, f1
    df = df.merge(dfs_dict["comics-novels"], how="inner", on=indexs)
    df = df.rename(
        columns={
            "sim_mode": "Representation",
            "use_weights": "Measure",
            "alignment": "Alignment",
            "f1_x": "Novels vs. TV Show",
            "f1_y": "Comics vs. TV Show",
            "f1": "Novels vs. Comics",
        }
    )
    df = df.set_index(["Representation", "Measure", "character_filtering", "Alignment"])
    # sort alignment so that thresholding comes first
    df = df.sort_values(by="Alignment", ascending=False)
    # fix the order of pairs
    df = df[["Novels vs. Comics", "Novels vs. TV Show", "Comics vs. TV Show"]]
    # put the "character filtering" level in the column
    df = df.unstack(2)
    # fix the order of the character sets
    df = df.reindex(["named", "common", "top-20"], axis=1, level=1)

    if args.best:
        df = df.groupby("Alignment").max()

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

    column_format = "lllrrrrrrrrr"
    if args.best:
        column_format = "l r@{~}r@{~}r r@{~}r@{~}r r@{~}r@{~}r"

    LaTeX_export = (
        df.style.format(lambda v: "{:.2f}".format(v * 100))
        .map_index(format_index, axis="columns")
        .highlight_max(subset="Novels vs. Comics", props="bfseries: ;", axis=None)
        .highlight_max(subset="Novels vs. TV Show", props="bfseries: ;", axis=None)
        .highlight_max(subset="Comics vs. TV Show", props="bfseries: ;", axis=None)
        .to_latex(
            hrules=True,
            sparse_index=True,
            multicol_align="c",
            multirow_align="c",
            clines="skip-last;data",
            column_format=column_format,
        )
    )
    # HACK: things we can't do with df.style.format.to_latex...
    # the last \cline does not make sense with \bottomline
    LaTeX_export = LaTeX_export.replace(
        "\\cline{1-12} \\cline{2-12}\n\\bottomrule", "\\bottomrule"
    )
    # also, there is a double line somewhere not making sense
    LaTeX_export = LaTeX_export.replace("\\cline{1-12} \\cline{2-12}", "\\cline{1-12}")
    # \cmidrule leaves more space between lines, it's more esthetic
    LaTeX_export = LaTeX_export.replace("cline", "cmidrule")
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
    #                                Thresholding             0.296053  0.347439  ...
    #
    # ... and it's the only way I found! Sorry!
    lines = LaTeX_export.split("\n")
    LEVELS_TITLES_I = 4
    TITLES_NB = 1 if args.best else 3
    levels_titles = lines[LEVELS_TITLES_I]
    titles = levels_titles.split("&")[:TITLES_NB]
    lines[2] = "&".join(titles) + "& " + lines[2][lines[2].index("\\") :]
    lines = lines[:LEVELS_TITLES_I] + lines[LEVELS_TITLES_I + 1 :]
    LaTeX_export = "\n".join(lines)

    print(LaTeX_export)
