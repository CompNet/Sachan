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
        "-m",
        "--medias",
        type=str,
        help="Either 'comics-novels', 'tvshow-comics' or 'tvshow-novels'",
    )
    parser.add_argument(
        "-a",
        "--alignment",
        type=str,
        default="threshold",
        help="one of 'threshold', 'smith-waterman'",
    )
    args = parser.parse_args()

    with open(
        f"{root_dir}/out/matching/plot/{args.medias}_structural/df.pickle", "rb"
    ) as f:
        df = pickle.load(f)

    # select
    df = df[df["alignment"] == args.alignment]

    # rearrange
    df = df.loc[:, ["f1", "sim_mode", "use_weights", "character_filtering"]]
    df = df.set_index(["sim_mode", "use_weights", "character_filtering"])
    df = df["f1"].unstack(["sim_mode", "use_weights"])

    # esthetic changes
    df.columns.names = ["Jaccard index", "weighted"]
    df.index.name = "character filtering"
    # level 1 here representends the "weighted" index
    df.columns = df.columns.set_levels(
        df.columns.levels[1].map({True: "yes", False: "no"}), level=1
    )
    df = df.reindex(["none", "common", "named", "common+named"])

    if args.format == "plain":
        print(df)
        exit(0)

    LaTeX_export = (
        df.style.format(lambda v: "{:.2f}".format(v * 100))
        .highlight_max(props="bfseries: ;", axis=None)
        .to_latex(hrules=True, sparse_index=False, multicol_align="c")
    )
    print(LaTeX_export)
