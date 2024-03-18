# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:07:13 2023

@author: shane mannion
Goes through the book datasets and creates networks, 
using names from the conversion maps
"""

import networkx as nx
import pandas as pd
from build_network import build_network, relabeled_with_id
#nodesp[node].get(atr, default)
if __name__ == '__main__':
    
    max_chapters = {'1.AGoT':72, '2.ACoK':69, '3.ASoS': 81, '4.AFFC': 45, '5.ADwD':72}
    G_cumulative = nx.Graph()
    for book in list(max_chapters.keys()):
        for i in range(0, max_chapters[book] + 1):
            G_instant = build_network(f'in/novels/raw/{book}',nx.Graph(),ch_start=i, ch_end=i)
            G_cumulative = build_network(f'in/novels/raw/{book}',G_cumulative,ch_start=i, ch_end=i)
            nx.set_node_attributes(G_instant, G_cumulative.nodes())
            H_instant = relabeled_with_id(G_instant, 'name')
            H_cumulative = relabeled_with_id(G_cumulative, 'name')
            if i < 10:
                nx.write_graphml(H_instant, f'in/novels/instant/chapter/{book}_0{i}_instant.graphml')
                nx.write_graphml(H_cumulative, f'in/novels/cumul/chapter/{book}_0{i}_cumul.graphml')
            else:
                nx.write_graphml(H_instant, f'in/novels/instant/chapter/{book}_{i}_instant.graphml')
                nx.write_graphml(H_cumulative, f'in/novels/cumul/chapter/{book}_{i}_cumul.graphml')       
