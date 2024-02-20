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

    dfs = {}

    for medias in ["tvshow-novels", "comics-novels"]:
        with open(
            f"{root_dir}/out/matching/plot/{medias}_structural_blocks/df.pickle", "rb"
        ) as f:
            df = pickle.load(f)
            dfs[medias] = df

    df = pd.DataFrame(
        {
            medias: [
                df[df["alignment"] == "threshold"]["f1"].max(axis=None),
                df[df["alignment"] == "smith-waterman"]["f1"].max(axis=None),
            ]
            for medias, df in dfs.items()
        },
        index=["threshold", "smith-waterman"],
    )
    df = df.rename(
        columns={
            "tvshow-novels": "Novels vs. TV Show",
            "comics-novels": "Novels vs. Comics",
        }
    )
    df = df[["Novels vs. Comics", "Novels vs. TV Show"]]

    if args.format == "plain":
        print(df)
        exit(0)

    LaTeX_export = (
        df.style.format(lambda v: "{:.2f}".format(v * 100))
        .map_index(lambda v: "itshape: ;", axis="columns")
        .highlight_max(props="bfseries: ;", axis=0)
        .to_latex(hrules=True, column_format="lcc")
    )
    print(LaTeX_export)
