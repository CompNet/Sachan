import os, pickle
import numpy as np
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
    combined_similarities,
    tune_alpha_other_medias,
)
from smith_waterman import (
    tune_smith_waterman_params_other_medias,
    smith_waterman_align_affine_gap,
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

        _, _, f1, _ = precision_recall_fscore_support(
            G_season.flatten(),
            M_season.flatten(),
            average="binary",
            zero_division=0.0,
        )
        season_f1s.append(f1)

    return season_f1s


if __name__ == "__main__":
    best_rows = {}

    for similarity in ["structural", "textual", "combined"]:
        with open(
            f"{root_dir}/out/matching/plot/tvshow-novels_{similarity}/df.pickle",
            "rb",
        ) as f:
            df = pickle.load(f)
            best_rows[similarity] = df.iloc[df["f1"].idxmax()]

    first_media_graphs, second_media_graphs = load_medias_graphs(
        "tvshow-novels", 1, 5, 1, 5
    )

    episode_summaries = load_tvshow_episode_summaries(1, 5)
    chapter_summaries = load_novels_chapter_summaries(1, 5)

    G = load_medias_gold_alignment("tvshow-novels", 1, 5, 1, 5)

    # Structural
    # ----------
    structural_params = best_rows["structural"]

    S = graph_similarity_matrix(
        first_media_graphs,
        second_media_graphs,
        structural_params["mode"],
        structural_params["use_weights"],
        structural_params["character_filtering"],
    )

    if structural_params["alignment"] == "threshold":
        t = tune_threshold_other_medias(
            "tvshow-novels",
            "structural",
            np.arange(0.0, 1.0, 0.01),
            structural_mode=structural_params["mode"],
            structural_use_weights=structural_params["use_weights"],
            structural_filtering=structural_params["character_filtering"],
        )
        M = S > t
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
            structural_mode=structural_params["mode"],
            structural_use_weights=structural_params["use_weights"],
            structural_filtering=structural_params["character_filtering"],
        )
        M, *_ = smith_waterman_align_affine_gap(
            S, gap_start_penalty, gap_cont_penalty, neg_th
        )
    else:
        raise ValueError

    season_f1s = compute_season_f1s(M, G)

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
        M = S > t
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
        M, *_ = smith_waterman_align_affine_gap(
            S,
            gap_start_penalty=gap_start_penalty,
            gap_cont_penalty=gap_cont_penalty,
            neg_th=neg_th,
        )
    else:
        raise ValueError

    season_f1s = compute_season_f1s(M, G)

    # Combined
    # --------
    combined_params = best_rows["combined"]

    S_textual = textual_similarity(
        episode_summaries, chapter_summaries, combined_params["sim_fn"]
    )

    S_structural = graph_similarity_matrix(
        first_media_graphs,
        second_media_graphs,
        combined_params["mode"],
        combined_params["use_weigths"],
        combined_params["character_filtering"],
    )

    if combined_params["combined"] == "threshold":
        alpha, t = tune_alpha_other_medias(
            "tvshow-novels",
            "threshold",
            np.arange(0.0, 1.0, 0.01),  # alpha
            [np.arange(0.0, 1.0, 0.01)],  # threshold
            textual_sim_fn=combined_params["sim_fn"],
            structural_mode=combined_params["mode"],
            structural_use_weights=combined_params["use_weights"],
            structural_filtering=combined_params["character_filtering"],
            silent=True,
        )
        S_combined = combined_similarities(S_structural, S_textual, alpha)
        M = S_combined > t
    elif combined_params["combined"] == "smith-waterman":
        (
            alpha,
            gap_start_penalty,
            gap_cont_penalty,
            neg_th,
        ) = tune_alpha_other_medias(
            "tvshow-novels",
            "smith-waterman",
            np.arange(0.1, 0.9, 0.05),  # alpha
            [
                np.arange(0.0, 0.2, 0.01),  # gap_start_penalty
                np.arange(0.0, 0.2, 0.01),  # gap_cont_penalty
                np.arange(0.0, 0.1, 0.1),  # neg_th
            ],
            textual_sim_fn=combined_params["sim_fn"],
            structural_mode=combined_params["mode"],
            structural_use_weights=combined_params["use_weigths"],
            structural_filtering=combined_params["character_filtering"],
            silent=True,
        )
        S_combined = combined_similarities(S_structural, S_textual, alpha)
        M, *_ = smith_waterman_align_affine_gap(
            S_combined, gap_start_penalty, gap_cont_penalty, neg_th
        )
    else:
        raise ValueError
