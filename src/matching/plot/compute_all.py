#!/usr/bin/python3
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../../out/matching/plot"


configurations = {
    "tvshow-novels": {"media_bounds": [(1, 6, 1, 5), (1, 2, 1, 2)]},
    "tvshow-comics": {"media_bounds": [(1, 2, 1, 2)]},
    "novels-comics": {"media_bounds": [(1, 2, 1, 2)]},
}

for config_name, config_dict in configurations.items():

    out_dir = f"{root_dir}/{config_name}"
    os.makedirs(out_dir, exist_ok=True)

    for media_bounds in config_dict["media_bounds"]:

        m1_start, m1_end, m2_start, m2_end = media_bounds

        # PERFORMANCE TABLES
        # ------------------
        # semantic performance table
        if config_name == "tvshow-novels":
            for sim_fn in ("sbert", "tfidf"):
                out_file = f"{out_dir}/perf_semantic_{sim_fn}_({m1_start}-{m1_end})_({m2_start}-{m2_end}).txt"
                command = f"python3 compute_semantic_alignment_performance.py -f plain -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} > '{out_file}'"
                print(command)
                os.system(command)

        # structural performance table
        out_file = (
            f"{out_dir}/perf_structural_({m1_start}-{m1_end})_({m2_start}-{m2_end}).txt"
        )
        command = f"python3 compute_structural_alignment_performance.py -f plain --medias '{config_name}' -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} > '{out_file}'"
        print(command)
        os.system(command)

        # combined performance table
        if config_name == "tvshow-novels":
            for sim_fn in ("sbert", "tfidf"):
                out_file = f"{out_dir}/perf_combined_{sim_fn}_({m1_start}-{m1_end})_({m2_start}-{m2_end}).txt"
                command = f"python3 compute_combined_alignment_performance.py -s {sim_fn} -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} > '{out_file}'"
                print(command)
                os.system(command)

        # GOLD ALIGNMENT PLOTS
        # --------------------
        out_file = f"{out_dir}/gold_({m1_start}-{m1_end})_({m2_start}-{m2_end}).pdf"
        command = f"python3 plot_gold_alignment.py --medias '{config_name}' -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} --output '{out_file}'"
        print(command)
        os.system(command)

        # PERFORMANCE THROUGH TIME
        # ------------------------
        # semantic performance table
        if config_name == "tvshow-novels":
            for sim_fn in ("sbert", "tfidf"):
                out_file = f"{out_dir}/perf_semantic_tt_{sim_fn}_({m1_start}-{m1_end})_({m2_start}-{m2_end}).pdf"
                command = f"python3 plot_semantic_alignment_perf_through_time.py -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} --output '{out_file}'"
                print(command)
                os.system(command)

        # structural performance table
        if config_name in ("tvshow-novels", "tvshow-comics"):
            out_file = f"{out_dir}/perf_structural_tt_({m1_start}-{m1_end})_({m2_start}-{m2_end}).pdf"
            command = f"python3 plot_structural_alignment_perf_through_time.py --medias '{config_name}' -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} --output '{out_file}'"
            print(command)
            os.system(command)

        # combined performance table
        if config_name == "tvshow-novels":
            for sim_fn in ("sbert", "tfidf"):
                out_file = f"{out_dir}/perf_combined_tt_{sim_fn}_({m1_start}-{m1_end})_({m2_start}-{m2_end}).pdf"
                command = f"python3 plot_combined_alignment_perf_through_time.py -s {sim_fn} -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} --output '{out_file}'"
                print(command)
                os.system(command)

        # PREDICTED ALIGNMENT
        # -------------------
        # semantic performance table
        if config_name == "tvshow-novels":
            for sim_fn in ("sbert", "tfidf"):
                out_file = f"{out_dir}/alignment_semantic_{sim_fn}_({m1_start}-{m1_end})_({m2_start}-{m2_end}).pdf"
                command = f"python3 plot_semantic_alignment.py -s {sim_fn} -t -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} --output '{out_file}'"
                print(command)
                os.system(command)

        # structural performance table
        out_file = f"{out_dir}/alignment_structural_({m1_start}-{m1_end})_({m2_start}-{m2_end}).pdf"
        command = f"python3 plot_structural_alignment.py --medias '{config_name}' -t -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} --output '{out_file}'"
        print(command)
        os.system(command)

        # combined performance table
        if config_name == "tvshow-novels":
            for sim_fn in ("sbert", "tfidf"):
                out_file = f"{out_dir}/alignment_combined_{sim_fn}_({m1_start}-{m1_end})_({m2_start}-{m2_end}).pdf"
                command = f"python3 plot_combined_alignment.py -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} --output '{out_file}'"
                print(command)
                os.system(command)
