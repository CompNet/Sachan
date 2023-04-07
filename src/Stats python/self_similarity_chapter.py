# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:36:01 2023

@author: shane
"""

import os
import networkx as nx
import pandas as pd
import csv
import numpy as np
import netrd
from HIM import Hamming, IM, gamma, gamma_s
from itertools import combinations
from build_network import build_network

dist = netrd.distance.NonBacktrackingSpectral()

max_chapters = {'1.AGoT':72, '2.ACoK':69, '3.ASoS': 81, '4.AFFC': 45, '5.ADwD':72}
G_cumulative = nx.Graph()
networks = []
for book in list(max_chapters.keys()):
    for i in range(0, max_chapters[book] + 1):
        G_instant = build_network(f'data/books/{book}',nx.Graph(),ch_start=i, ch_end=i)
        networks.append(G_instant)
     
df = pd.DataFrame()

for number, chapter in enumerate(networks):
    distances = []
    for chapter_b in networks:
        if len(chapter.edges()) == 0 or len(chapter_b.edges()) == 0:
            distances.append('NaN')
        else:
            distances.append(IM(chapter, chapter_b, gamma_s(chapter, chapter_b)))

    df[f'chapter_{number}'] = distances
    
df.to_csv('self_similarity_NB.csv', index=False)
    