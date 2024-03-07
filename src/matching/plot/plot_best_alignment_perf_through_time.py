import os, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_recall_fscore_support
from alignment_commons import (
    tune_threshold_other_medias,
    graph_similarity_matrix,
    load_medias_graphs,
    load_tvshow_episode_summaries,
    load_novels_chapter_summaries,
    TVSHOW_SEASON_LIMITS,
    load_medias_gold_alignment,
    textual_similarity,
    tune_alpha_other_medias,
    align_blocks,
    combined_similarities_blocks,
)
from smith_waterman import (
    tune_smith_waterman_params_other_medias,
    smith_waterman_align_affine_gap,
    smith_waterman_align_blocks,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."


def compute_season_f1s(M: np.ndarray, G: np.ndarray) -> list[float]:
    season_f1s = []

    for season in range(1, 6):
        limits = [0] + TVSHOW_SEASON_LIMITS
        start = limits[season - 1]
        end = limits[season]
        G_season = G[start:end, :]
        M_season = M[start:end, :]

        f1, _ = precision_recall_fscore_support(
            G_season.flatten(),
            M_season.flatten(),
            average="binary",
            zero_division=0.0,
        )
        season_f1s.append(f1)

    return season_f1s


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    best_rows = {}

    for similarity in ["structural", "textual", "combined"]:
        if similarity == ["structural", "combined"]:
            # blocks have better performance for structural and combined
            # matching
            path = f"{root_dir}/out/matching/plot/tvshow-novels_{similarity}_blocks/df.pickle"
        else:
            path = f"{root_dir}/out/matching/plot/tvshow-novels_{similarity}/df.pickle"
        with open(path, "rb") as f:
            df = pickle.load(f)
            best_rows[similarity] = df.iloc[df["f1"].idxmax()]

    first_media_graphs, second_media_graphs = load_medias_graphs(
        "tvshow-novels", 1, 5, 1, 5
    )

    episode_summaries = load_tvshow_episode_summaries(1, 5)
    chapter_summaries = load_novels_chapter_summaries(1, 5)

    G = load_medias_gold_alignment("tvshow-novels", 1, 5, 1, 5)

    plt.style.use("science")
    fig, ax = plt.subplots()
    ax.set_xlabel("Seasons")
    ax.set_ylabel("F1")
    ax.grid()

    # Structural
    # ----------
    structural_params = best_rows["structural"]

    S = graph_similarity_matrix(
        first_media_graphs,
        second_media_graphs,
        structural_params["sim_mode"],
        structural_params["use_weights"],
        structural_params["character_filtering"],
    )

    if structural_params["alignment"] == "threshold":
        t = tune_threshold_other_medias(
            "tvshow-novels",
            "structural",
            np.arange(0.0, 1.0, 0.01),
            structural_mode=structural_params["sim_mode"],
            structural_use_weights=structural_params["use_weights"],
            structural_filtering=structural_params["character_filtering"],
        )
        M_structural = align_blocks(
            "tvshow-novels", first_media_graphs, second_media_graphs, S > t
        )
    elif structural_params["alignment"] == "smith-waterman":
        (
            gap_start_penalty,
            gap_cont_penalty,
            neg_th,
        ) = tune_smith_waterman_params_other_medias(
            "tvshow-novels",
            "structural",
            np.arange(0.0, 0.2, 0.01),
            np.arange(0.0, 0.2, 0.01),
            np.arange(0.0, 0.1, 0.1),
            structural_mode=structural_params["sim_mode"],
            structural_use_weights=structural_params["use_weights"],
            structural_filtering=structural_params["character_filtering"],
        )
        M_structural = smith_waterman_align_blocks(
            "tvshow-novels",
            first_media_graphs,
            second_media_graphs,
            S,
            gap_start_penalty=gap_start_penalty,
            gap_cont_penalty=gap_cont_penalty,
            neg_th=neg_th,
        )
    else:
        raise ValueError

    season_f1s = compute_season_f1s(M_structural, G)
    ax.plot(list(range(1, 6)), season_f1s, label="Structural")

    # Textual
    # -------
    textual_params = best_rows["textual"]

    S = textual_similarity(
        episode_summaries, chapter_summaries, textual_params["sim_fn"]
    )

    if textual_params["alignment"] == "threshold":
        t = tune_threshold_other_medias(
            "tvshow-novels",
            "textual",
            np.arange(0.0, 1.0, 0.01),
            textual_sim_fn=textual_params["sim_fn"],
            silent=True,
        )
        M_textual = S > t
    elif textual_params["alignment"] == "smith-waterman":
        (
            gap_start_penalty,
            gap_cont_penalty,
            neg_th,
        ) = tune_smith_waterman_params_other_medias(
            "tvshow-novels",
            "textual",
            np.arange(0.0, 0.2, 0.01),
            np.arange(0.0, 0.2, 0.01),
            np.arange(0.0, 0.1, 0.1),
            textual_sim_fn=textual_params["sim_fn"],
            silent=True,
        )
        M_textual, *_ = smith_waterman_align_affine_gap(
            S,
            gap_start_penalty=gap_start_penalty,
            gap_cont_penalty=gap_cont_penalty,
            neg_th=neg_th,
        )
    else:
        raise ValueError

    season_f1s = compute_season_f1s(M_textual, G)
    ax.plot(list(range(1, 6)), season_f1s, label="Textual")

    # Combined
    # --------
    combined_params = best_rows["combined"]

    S_textual = textual_similarity(
        episode_summaries, chapter_summaries, combined_params["textual_sim_fn"]
    )

    S_structural = graph_similarity_matrix(
        first_media_graphs,
        second_media_graphs,
        combined_params["structural_sim_mode"],
        combined_params["structural_use_weights"],
        combined_params["structural_character_filtering"],
    )

    if combined_params["alignment"] == "threshold":
        alpha, t = tune_alpha_other_medias(
            "tvshow-novels",
            "threshold",
            np.arange(0.0, 1.0, 0.01),  # alpha
            [np.arange(0.0, 1.0, 0.01)],  # threshold
            textual_sim_fn=combined_params["textual_sim_fn"],
            structural_mode=combined_params["structural_sim_mode"],
            structural_use_weights=combined_params["structural_use_weights"],
            structural_filtering=combined_params["structural_character_filtering"],
            silent=True,
        )
        S_combined = combined_similarities_blocks(
            S_structural,
            S_textual,
            alpha,
            "tvshow-novels",
            first_media_graphs,
            second_media_graphs,
        )
        M_combined = align_blocks(
            "tvshow-novels", first_media_graphs, second_media_graphs, S_combined > t
        )
    elif combined_params["alignment"] == "smith-waterman":
        (alpha, gap_start_penalty, gap_cont_penalty, neg_th,) = tune_alpha_other_medias(
            "tvshow-novels",
            "smith-waterman",
            np.arange(0.1, 0.9, 0.05),  # alpha
            [
                np.arange(0.0, 0.2, 0.01),  # gap_start_penalty
                np.arange(0.0, 0.2, 0.01),  # gap_cont_penalty
                np.arange(0.0, 0.1, 0.1),  # neg_th
            ],
            textual_sim_fn=combined_params["textual_sim_fn"],
            structural_mode=combined_params["structural_sim_mode"],
            structural_use_weights=combined_params["structural_use_weights"],
            structural_filtering=combined_params["structural_character_filtering"],
            silent=True,
        )
        S_combined = combined_similarities_blocks(
            S_structural,
            S_textual,
            alpha,
            "tvshow-novels",
            first_media_graphs,
            second_media_graphs,
        )
        M_combined = smith_waterman_align_blocks(
            "tvshow-novels",
            first_media_graphs,
            second_media_graphs,
            S_combined,
            gap_start_penalty=gap_start_penalty,
            gap_cont_penalty=gap_cont_penalty,
            neg_th=neg_th,
        )
    else:
        raise ValueError

    season_f1s = compute_season_f1s(M_combined, G)
    ax.plot(list(range(1, 6)), season_f1s, label="Combined")

    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
