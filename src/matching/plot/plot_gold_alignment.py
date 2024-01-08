# Plot gold alignment between two medias (episodes / chapters granularity).
#
#
# Example usage:
#
# python plot_gold_alignment.py -m 'tvshow-novels'
#
#
# For more details, see:
#
# python plot_gold_alignment.py --help
#
#
# Author: Arthur Amalvy
import argparse
import scienceplots
import matplotlib.pyplot as plt
from alignment_commons import load_medias_gold_alignment

if __name__ == "__main__":

    FONTSIZE = 10
    COLUMN_WIDTH_IN = 5.166

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--medias",
        type=str,
        help="Medias on which to compute alignment. Either 'comics-novels', 'tvshow-comics' or 'tvshow-novels'",
    )
    parser.add_argument("-m1", "--min-delimiter-first-media", type=int, default=None)
    parser.add_argument("-x1", "--max-delimiter-first-media", type=int, default=None)
    parser.add_argument("-m2", "--min-delimiter-second-media", type=int, default=None)
    parser.add_argument("-x2", "--max-delimiter-second-media", type=int, default=None)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    G = load_medias_gold_alignment(
        args.medias,
        args.min_delimiter_first_media,
        args.max_delimiter_first_media,
        args.min_delimiter_second_media,
        args.max_delimiter_second_media,
    )

    plt.style.use("science")
    fig, ax = plt.subplots()
    fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)
    ax.imshow(G, interpolation="none")

    if args.medias == "comics-novels":
        ax.set_ylabel("Comics Issues", fontsize=FONTSIZE)
        ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)
    elif args.medias == "tvshow-comics":
        ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
        ax.set_xlabel("Comics Issues", fontsize=FONTSIZE)
    elif args.medias == "tvshow-novels":
        ax.set_ylabel("TV Show Episodes", fontsize=FONTSIZE)
        ax.set_xlabel("Novels Chapters", fontsize=FONTSIZE)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()
