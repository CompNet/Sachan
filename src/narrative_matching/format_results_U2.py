import argparse, os, pickle
import pandas as pd


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../.."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--format", type=str, default="latex", help="either 'plain' or 'latex'"
    )
    args = parser.parse_args()

    with open(
        f"{root_dir}/out/narrative_matching/tvshow-novels_structural_U2/df.pickle", "rb"
    ) as f:
        df = pickle.load(f)
        best_struct_f1_th = df[df["alignment"] == "threshold"]["f1"].max()
        best_struct_f1_sw = df[df["alignment"] == "smith-waterman"]["f1"].max()

    with open(
        f"{root_dir}/out/narrative_matching/tvshow-novels_structural_blocks_U2/df.pickle",
        "rb",
    ) as f:
        df = pickle.load(f)
        best_struct_blocks_f1_th = df[df["alignment"] == "threshold"]["f1"].max()
        best_struct_blocks_f1_sw = df[df["alignment"] == "smith-waterman"]["f1"].max()

    with open(
        f"{root_dir}/out/narrative_matching/tvshow-novels_textual_U2/df.pickle", "rb"
    ) as f:
        df = pickle.load(f)
        best_text_f1_th = df[df["alignment"] == "threshold"]["f1"].max()
        best_text_f1_sw = df[df["alignment"] == "smith-waterman"]["f1"].max()

    with open(
        f"{root_dir}/out/narrative_matching/tvshow-novels_combined_U2/df.pickle", "rb"
    ) as f:
        df = pickle.load(f)
        best_combined_f1_th = df[df["alignment"] == "threshold"]["f1"].max()
        best_combined_f1_sw = df[df["alignment"] == "smith-waterman"]["f1"].max()

    with open(
        f"{root_dir}/out/narrative_matching/tvshow-novels_combined_blocks_U2/df.pickle",
        "rb",
    ) as f:
        df = pickle.load(f)
        best_combined_blocks_f1_th = df[df["alignment"] == "threshold"]["f1"].max()
        best_combined_blocks_f1_sw = df[df["alignment"] == "smith-waterman"]["f1"].max()

    df = pd.DataFrame(
        [
            [
                best_text_f1_th,
                best_struct_f1_th,
                best_struct_blocks_f1_th,
                best_combined_f1_th,
                best_combined_blocks_f1_th,
            ],
            [
                best_text_f1_sw,
                best_struct_f1_sw,
                best_struct_blocks_f1_sw,
                best_combined_f1_sw,
                best_combined_blocks_f1_sw,
            ],
        ],
        columns=[
            "Textual",
            "Structural",
            "Structural (+sub-units)",
            "Hybrid",
            "Hybrid (+sub-units)",
        ],
        index=["Thresholding", "Smith--Waterman"],
    )
    df.index.name = "Alignment"

    if args.format == "plain":
        print(df)
        exit(0)

    LaTeX_export = (
        df.style.format(lambda v: "{:.2f}".format(v * 100))
        .highlight_max(props="bfseries: ;", axis=None)
        .to_latex(hrules=True, sparse_index=False)
    )
    print(LaTeX_export)
