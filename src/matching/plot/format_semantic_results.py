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
        "-a",
        "--alignment",
        type=str,
        default="threshold",
        help="one of 'threshold', 'smith-waterman'",
    )
    args = parser.parse_args()

    dfs = {}

    for medias in ["tvshow-novels", "tvshow-comics", "comics-novels"]:
        with open(
            f"{root_dir}/out/matching/plot/{medias}_semantic/df.pickle", "rb"
        ) as f:
            df = pickle.load(f)
            dfs[medias] = df[df["alignment"] == args.alignment]

    df_dict = {
        medias: df.loc[:, ["sim_fn", "f1"]].set_index("sim_fn")
        for medias, df in dfs.items()
    }

    # output cols: sim_fn, f1_x, f1_y
    df = df_dict["tvshow-novels"].merge(
        df_dict["tvshow-comics"], how="inner", on="sim_fn"
    )
    # output cols: sim_fn, f1_x, f1_y, f1
    df = df.merge(df_dict["comics-novels"], how="inner", on="sim_fn")
    df = df.rename(
        columns={
            "f1_x": "tvshow-novels",
            "f1_y": "tvshow-comics",
            "f1": "comics-novels",
        }
    )
    df.index = df.index.rename("similarity")

    if args.format == "plain":
        print(df)
        exit(0)

    LaTeX_export = (
        df.style.format(lambda v: "{:.2f}".format(v * 100))
        .highlight_max(props="bfseries: ;", axis=0)
        .to_latex(hrules=True)
    )
    print(LaTeX_export)
