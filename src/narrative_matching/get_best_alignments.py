import os, pickle, argparse
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../.."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--no-blocks",
        action="store_true",
        help="if specified, does not include blocks results.",
    )
    args = parser.parse_args()

    for medias, period in [
        ("tvshow-novels", ""),
        ("tvshow-novels", "_U2"),
        ("tvshow-comics", ""),
        ("comics-novels", ""),
    ]:
        best_rows = []

        for similarity in ["structural", "textual", "combined"]:
            with open(
                f"{root_dir}/out/narrative_matching/{medias}_{similarity}{period}/df.pickle",
                "rb",
            ) as f:
                df = pickle.load(f)
                best_row = df.iloc[df["f1"].idxmax()]
                additional_datas = pd.Series(
                    [similarity, False], index=["similarity", "blocks"]
                )
                best_rows.append(pd.concat([additional_datas, best_row]))

            if args.no_blocks:
                continue

            # blocks
            if similarity in ["structural", "combined"]:
                with open(
                    f"{root_dir}/out/narrative_matching/{medias}_{similarity}_blocks{period}/df.pickle",
                    "rb",
                ) as f:
                    df = pickle.load(f)
                    best_row = df.iloc[df["f1"].idxmax()]
                    additional_datas = pd.Series(
                        [similarity, True], index=["similarity", "blocks"]
                    )
                    best_rows.append(pd.concat([additional_datas, best_row]))

        overall_best_row = max(best_rows, key=lambda r: r["f1"])
        print(f"best configuration for {medias}{period}: ")
        print(overall_best_row, end="\n\n")
