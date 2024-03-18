# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:03:44 2023

@author: shane
"""
import os
import pandas as pd
from summary_stats import summary_stats

##########
#Book instant chapter:
dir_path = 'in/novels/instant/'

##########
#Datasets
networks = []
for file in os.listdir(dir_path):
    networks.append(f'{dir_path}/{file}')
    
#print(networks)
networks.sort()

df = summary_stats(networks, 'book', 'out/descript/summary/summary_stats_books_instant_chapter')

#Book cumulative chapter:
dir_path = 'in/novels/cumul/'

##########
#Datasets
networks = []
for file in os.listdir(dir_path):
    networks.append(f'{dir_path}/{file}')
    
#print(networks)
networks.sort()

df = summary_stats(networks, 'book', 'out/descript/summary/summary_stats_books_cumulative_chapter')


#Comic instant chapter:
dir_path = 'in/comics/instant/chapter/'

##########
#Datasets
networks = []
for file in os.listdir(dir_path):
    networks.append(f'{dir_path}/{file}')
    
#print(networks)
networks.sort()

df = summary_stats(networks, 'comic', 'out/descript/summary/summary_stats_comics_instant_chapter')


#Comic cumulative chapter:
dir_path = 'in/comics/cumul/chapter/'

##########
#Datasets
networks = []
for file in os.listdir(dir_path):
    networks.append(f'{dir_path}/{file}')
    
#print(networks)
networks.sort()

df = summary_stats(networks, 'comic', 'out/descript/summary/summary_stats_comics_cumulative_chapter')

#Comic instant scene:
dir_path = 'in/comics/instant/scene/'

##########
#Datasets
networks = []
for file in os.listdir(dir_path):
    networks.append(f'{dir_path}/{file}')
    
#print(networks)
networks.sort()

df = summary_stats(networks, 'comic', 'out/descript/summary/summary_stats_comics_instant_scene')


#Comic cumulative scene:
dir_path = 'in/comics/cumul/scene/'

##########
#Datasets
networks = []
for file in os.listdir(dir_path):
    networks.append(f'{dir_path}/{file}')
    
#print(networks)
networks.sort()

df = summary_stats(networks, 'comic', 'out/descript/summary/summary_stats_comics_cumulative_scene')


#Show instant episode:
dir_path = 'in/tvshow/instant/episode/'

##########
#Datasets
networks = []
for file in os.listdir(dir_path):
    networks.append(f'{dir_path}/{file}')
    
#print(networks)
networks.sort()

df = summary_stats(networks, 'show', 'out/descript/summary/summary_stats_show_instant_episode')

#Show cumulative episode:
dir_path = 'in/tvshow/cumul/episode/'

##########
#Datasets
networks = []
for file in os.listdir(dir_path):
    networks.append(f'{dir_path}/{file}')
    
#print(networks)
networks.sort()

df = summary_stats(networks, 'show', 'out/descript/summary/summary_stats_show_cumulative_episode')

#Show instant scene:
dir_path = 'in/tvshow/instant/scene/'

##########
#Datasets
networks = []
for file in os.listdir(dir_path):
    networks.append(f'{dir_path}/{file}')
    
#print(networks)
networks.sort()

df = summary_stats(networks, 'show', 'out/descript/summary/summary_stats_show_instant_scene')

#Show cumulative episode:
dir_path = 'in/tvshow/cumul/scene/'

##########
#Datasets
networks = []
for file in os.listdir(dir_path):
    networks.append(f'{dir_path}/{file}')
    
#print(networks)
networks.sort()

df = summary_stats(networks, 'show', 'out/descript/summary/summary_stats_show_cumulative_scene')

