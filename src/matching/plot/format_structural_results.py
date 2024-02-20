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
    args = parser.parse_args()

    dfs_dict = {}
    for medias in ["tvshow-novels", "tvshow-comics", "comics-novels"]:
        with open(
            f"{root_dir}/out/matching/plot/{medias}_structural/df.pickle", "rb"
        ) as f:
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
            weighted_remap = {True: "yes", False: "no"}
            df["use_weights"] = df["use_weights"].apply(weighted_remap.get)
            dfs_dict[medias] = df

    indexs = ["sim_mode", "use_weights", "character_filtering", "alignment"]
    # output cols: sim_fn, f1_x, f1_y
    df = dfs_dict["tvshow-novels"].merge(
        dfs_dict["tvshow-comics"], how="inner", on=indexs
    )
    # output cols: sim_fn, f1_x, f1_y, f1
    df = df.merge(dfs_dict["comics-novels"], how="inner", on=indexs)
    df = df.rename(
        columns={
            "sim_mode": "Jaccard index",
            "use_weights": "weighted",
            "character_filtering": "character filtering",
            "f1_x": "Novels vs. TV Show",
            "f1_y": "Comics vs. TV Show",
            "f1": "Novels vs. Comics",
        }
    )
    df = df.set_index(["Jaccard index", "weighted", "character filtering", "alignment"])
    # put the "character filtering" level in the column
    df = df.unstack(2)

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
            sparse_index=False,
            multicol_align="c",
            column_format="lllccccccccc",
        )
    )
    print(LaTeX_export)
