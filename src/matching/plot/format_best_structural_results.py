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

    for medias in ["tvshow-novels", "tvshow-comics", "comics-novels"]:
        with open(
            f"{root_dir}/out/matching/plot/{medias}_structural/df.pickle", "rb"
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

    if args.format == "plain":
        print(df)
        exit(0)

    LaTeX_export = (
        df.style.format(lambda v: "{:.2f}".format(v * 100))
        .highlight_max(props="bfseries: ;", axis=0)
        .to_latex(hrules=True)
    )
    print(LaTeX_export)
