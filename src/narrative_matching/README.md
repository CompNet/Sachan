# What's here?

This directory contains all script relating to narrative matching. You can always obtain the arguments of a script using `python script.py --help`.


# Alignment Performance Computation

`compute_alignment_performance.py` computes the alignment performance of all configuration for a media pair and a similarity computation method (`structural`, `textual` of `combined`), while `compute_all.sh` use the previous script to compute performance for all media pairs and similarity computation methods. The results are saved as pickled dataframes under `out/matching/plot`.


# Gold Alignment

`plot_gold_alignment.py` can be used to plot the reference alignment for a specific media pair.


# Formatting Results Tables

All the python scripts starting with `format_` are used to format results obtained with the `compute_alignment_performance.py` script.

`plot_alignment.py` can be used to visualize a specific alignment configuration (see `--help` for the numerous options).


# Low-level Library

`alignment_commons.py`, `smith_waterman.py` and `graph_utils.py` form a library of many useful functions for performing narrative matching, used by the other scripts.
