#!/usr/bin/python3
from typing import Tuple
import os


def print_exec(command: str):
    print(command)
    os.system(command)


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../../out/matching/plot"


configurations = {
    "tvshow-novels": {"media_bounds": [(1, 6, 1, 5), (1, 2, 1, 2)]},
    "tvshow-comics": {"media_bounds": [(1, 2, 1, 2)]},
    "comics-novels": {"media_bounds": [(1, 2, 1, 2)]},
}

for config_name, config_dict in configurations.items():
    out_dir = f"{root_dir}/{config_name}"
    os.makedirs(out_dir, exist_ok=True)

    for media_bounds in config_dict["media_bounds"]:
        m1_start, m1_end, m2_start, m2_end = media_bounds

        def build_out_path(name: str, extension: str) -> str:
            m1_start, m1_end, m2_start, m2_end = media_bounds
            return f"{out_dir}/{name}_({m1_start}-{m1_end})_({m2_start}-{m2_end}).{extension}"

        def build_command(python_file: str, end: str) -> str:
            m1_start, m1_end, m2_start, m2_end = media_bounds
            return f"python3 {python_file} -m1 {m1_start} -x1 {m1_end} -m2 {m2_start} -x2 {m2_end} {end}"

        for alignment in ("threshold", "smith-waterman"):

            for similarity in ("structural", "semantic", "combined"):

                # PERFORMANCE TABLES
                # ------------------
                out_file = build_out_path(f"perf_{similarity}_{alignment}", "tex")
                command = build_command(
                    "compute_alignment_performance.py",
                    f"--medias '{config_name}' -s {similarity} -a {alignment} -f latex > '{out_file}'",
                )
                print_exec(command)

        # BLOCKS (TABLES ONLY)
        # --------------------
        if config_name == "tvshow-novels":
            out_file = build_out_path(f"perf_structural_{alignment}_blocks", "txt")
            command = build_command(
                "compute_alignment_performance.py",
                f"--blocks --medias '{config_name}' -s structural -a {alignment} -f latex > '{out_file}'",
            )
            print_exec(command)

        # GOLD ALIGNMENT PLOTS
        # --------------------
        out_file = build_out_path("gold", "pdf")
        command = build_command(
            "plot_gold_alignment.py", f"--medias '{config_name}' --output '{out_file}'"
        )
        print_exec(command)
