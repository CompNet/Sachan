#!/bin/bash

python3 compute_alignment_performance.py -m tvshow-novels -s structural
python3 compute_alignment_performance.py -m tvshow-novels -s textual
python3 compute_alignment_performance.py -m tvshow-novels -s combined

python3 compute_alignment_performance.py -m tvshow-comics -s structural
python3 compute_alignment_performance.py -m tvshow-comics -s textual
python3 compute_alignment_performance.py -m tvshow-comics -s combined

python3 compute_alignment_performance.py -m comics-novels -s structural
python3 compute_alignment_performance.py -m comics-novels -s textual
python3 compute_alignment_performance.py -m comics-novels -s combined

# blocks
python3 compute_alignment_performance.py -m tvshow-novels -s structural --blocks
python3 compute_alignment_performance.py -m comics-novels -s structural --blocks
