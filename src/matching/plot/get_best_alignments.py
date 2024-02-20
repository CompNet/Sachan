import os, pickle
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."

if __name__ == "__main__":
    for medias in ["tvshow-novels", "tvshow-comics", "comics-novels"]:
        best_rows = []

        for similarity in ["structural", "textual", "combined"]:
            with open(
                f"{root_dir}/out/matching/plot/{medias}_{similarity}/df.pickle", "rb"
            ) as f:
                df = pickle.load(f)
                best_row = df.iloc[df["f1"].idxmax()]
                best_rows.append(
                    pd.concat([pd.Series([similarity], index=["similarity"]), best_row])
                )

        overall_best_row = max(best_rows, key=lambda r: r["f1"])
        print(f"best configuration for {medias}: ")
        print(overall_best_row, end="\n\n")
