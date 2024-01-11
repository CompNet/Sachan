#!/bin/bash

python3 compute_performance.py -m tvshow-novels -s structural
python3 compute_performance.py -m tvshow-novels -s semantic
python3 compute_performance.py -m tvshow-novels -s combined

python3 compute_performance.py -m tvshow-comics -s structural
python3 compute_performance.py -m tvshow-comics -s semantic
python3 compute_performance.py -m tvshow-comics -s combined

python3 compute_performance.py -m comics-novels -s structural
python3 compute_performance.py -m comics-novels -s semantic
python3 compute_performance.py -m comics-novels -s combined
