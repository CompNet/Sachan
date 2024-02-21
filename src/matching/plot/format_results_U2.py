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

    with open(
        f"{root_dir}/out/matching/plot/tvshow-novels_structural_U2/df.pickle", "rb"
    ) as f:
        struct_df = pickle.load(f)
        best_struct_f1 = struct_df["f1"].max()

    with open(
        f"{root_dir}/out/matching/plot/tvshow-novels_structural_U2/df.pickle", "rb"
    ) as f:
        struct_df = pickle.load(f)
        best_struct_blocks_f1 = struct_df["f1"].max()

    with open(
        f"{root_dir}/out/matching/plot/tvshow-novels_textual_U2/df.pickle", "rb"
    ) as f:
        text_df = pickle.load(f)
        best_text_f1 = text_df["f1"].max()

    with open(
        f"{root_dir}/out/matching/plot/tvshow-novels_combined_U2/df.pickle", "rb"
    ) as f:
        combined_df = pickle.load(f)
        best_combined_f1 = combined_df["f1"].max()

    df = pd.DataFrame(
        [[best_struct_f1, best_struct_blocks_f1, best_text_f1, best_combined_f1]],
        columns=["structural", "structural (+sub-units)" "textual", "combined"],
    )

    if args.format == "plain":
        print(df)
        exit(0)

    LaTeX_export = (
        df.style.format(lambda v: "{:.2f}".format(v * 100))
        .highlight_max(props="bfseries: ;", axis=1)
        .hide(axis="index")
        .to_latex(hrules=True, sparse_index=False, column_format="ccc")
    )
    print(LaTeX_export)
