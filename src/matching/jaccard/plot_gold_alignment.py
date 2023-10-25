# Plot gold alignment (tvshow / novels)
import argparse, pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    CHAPTERS_NB = 344
    EPISODES_NB = 60

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gold-alignment",
        type=str,
        default=None,
        help="Path to the gold chapters/episodes alignment.",
    )
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.gold_alignment, "rb") as f:
        G = pickle.load(f)
    G = G[:EPISODES_NB, :CHAPTERS_NB]
    assert G.shape == (EPISODES_NB, CHAPTERS_NB)

    fig, ax = plt.subplots()
    fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)
    ax.imshow(G, interpolation="none")
    ax.set_xlabel("Chapters", fontsize=FONTSIZE)
    ax.set_ylabel("Episodes", fontsize=FONTSIZE)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
