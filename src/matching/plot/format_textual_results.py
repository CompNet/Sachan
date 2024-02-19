import argparse, os, pickle
from posixpath import split

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

    dfs = {}

    for medias in ["tvshow-novels", "tvshow-comics", "comics-novels"]:
        with open(
            f"{root_dir}/out/matching/plot/{medias}_textual/df.pickle", "rb"
        ) as f:
            dfs[medias] = pickle.load(f)

    df_dict = {
        medias: df.loc[:, ["sim_fn", "alignment", "f1"]] for medias, df in dfs.items()
    }

    # output cols: sim_fn, f1_x, f1_y
    df = df_dict["tvshow-novels"].merge(
        df_dict["tvshow-comics"], how="inner", on=["sim_fn", "alignment"]
    )
    # output cols: sim_fn, f1_x, f1_y, f1
    df = df.merge(df_dict["comics-novels"], how="inner", on=["sim_fn", "alignment"])
    df = df.rename(
        columns={
            "sim_fn": "similarity",
            "f1_x": "Novels vs. TV Show",
            "f1_y": "Comics vs. TV Show",
            "f1": "Novels vs. Comics",
        }
    )
    df = df[["Novels vs. Comics", "Novels vs. TV Show", "Comics vs. TV Show"]]

    df = df.set_index(["similarity", "alignment"])

    if args.format == "plain":
        print(df)
        exit(0)

    LaTeX_export = (
        df.style.format(lambda v: "{:.2f}".format(v * 100))
        .map_index(lambda v: "itshape: ;", axis="columns")
        .highlight_max(props="bfseries: ;", axis=0)
        .to_latex(hrules=True, column_format="llccc")
    )
    # HACK: add a rogue midrule in between multirow blocks
    # the header consists in \begin + \toprule + columns + index +
    # midrule + first multirow (2 lines)
    HEADER_SIZE = 7
    splitted = LaTeX_export.split("\n")
    splitted = splitted[:HEADER_SIZE] + ["\\midrule"] + splitted[HEADER_SIZE:]
    LaTeX_export = "\n".join(splitted)
    print(LaTeX_export)
