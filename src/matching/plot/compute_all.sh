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
python3 compute_alignment_performance.py -m tvshow-comics -s structural --blocks

# cumulative networks
python3 compute_alignment_performance.py -m tvshow-novels -s structural --cumulative
python3 compute_alignment_performance.py -m tvshow-comics -s structural --cumulative
python3 compute_alignment_performance.py -m comics-novels -s structural --cumulative

# tvshow-novels U2
python3 compute_alignment_performance.py -m tvshow-novels -s structural -p U2
python3 compute_alignment_performance.py -m tvshow-novels -s textual -p U2
python3 compute_alignment_performance.py -m tvshow-novels -s combined -p U2
python3 compute_alignment_performance.py -m tvshow-novels -s structural --blocks -p U2
