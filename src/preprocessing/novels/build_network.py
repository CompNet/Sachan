"""
# -*- coding: utf-8 -*-
#
# Created by Pádraig Mac Carron and Shane Mannion



#Read network
#
# This reads a tsv file.
# It identifies the headers initially so it can determine a character's gender and what type
# of link it is (i.e. friendly or hostile).
#
# As it reads the document it looks for chapters (or books or scenes) and if two characters
# are linked in two different chatpers it will increase the weight of that edge. 
#
# Chapter information is also added to the edge.
# If the argument "chapter_list=True" is added when the function is called
# it will output a list of chapters in the order they appear also as networkx
# will not keep the order.
#
# Start/end chapters can be added to create networks of sections of the story
#
# Then uses the name conversion maps to get the normalised names
#
# Some datasets have a faction as well, if that is to be 
# included then add the argument "Faction=True".
#
# Set "Hostile=False" if there are no hostile edges
#
"""

########################
#Import Libraries
import networkx as nx
import copy 
import pandas as pd
import math
########################
character_list = pd.read_excel('in/novels/raw/character_list.xlsx', 'MapNovels')
def relabeled_with_id(G: nx.Graph, attribute: str) -> nx.Graph:
    """Relabel a graph with unique number ids as nodes, and the current node as an attribute
    :param G:
    :param attribute:
    """
    H = copy.deepcopy(G)
    for node in H.nodes:
        H.nodes[node][attribute] = node
    H = nx.relabel_nodes(H, {node: i for i, node in enumerate(H.nodes())})
    return H


def get_normalised_name(novelname:str):
    
    name_row = character_list[character_list['NovelsName']==novelname]
    if len(name_row) == 0:
        print(f'Name not in character list: {novelname}')
        return novelname
    if type(name_row.iloc[0,2]) == float:
        if type(name_row.iloc[0,1]) == float:
            newname = novelname
        else:    
            newname = name_row.iloc[0,1]
        return newname
    else:
        if 'delete' in name_row.iloc[0,2].lower():
            newname = ''
            return newname
        if type(name_row.iloc[0,1]) == float:            
            newname = novelname
            return newname
        if 'merge' in name_row.iloc[0,2].lower():
            newname = name_row.iloc[0,1]
        if 'several' in name_row.iloc[0,2].lower():
            newname = name_row.iloc[0,1]

    return newname
    
        
    

        
def build_network(network,start_net=None, ch_start=0, ch_end=None, chapter_list=False,Faction=True,Hostile=True,file_type=".tsv"):
    max_chapters = {'AGoT':72, 'ACoK':69, 'ASoS': 81, 'AFFC': 45, 'ADwD':72}
    if ch_end == None:
        ch_end = max_chapters[network[-4:]]
        
    network = open(network + file_type,"r")

    if start_net == None:
        G = nx.Graph()
    else:
        G = start_net

    #Gets the headers and their associated index to call from later
    line = network.readline().rstrip()
    list_line = line.split('\t')
       
    headers = []
    for i in list_line:
        headers += [i]

    gender = -1
    pg = -1
    dead = -1
    for h in headers:
        if 'friendly link 1' == h.lower():
            friend = headers.index(h)
        if 'hostile link 1' == h.lower():
            hostile = headers.index(h)
        if 'gender' in h.lower():
            gender = headers.index(h)
        if 'page in' in h.lower():
            pg = headers.index(h)
        if 'page out' in h.lower():
            dead = headers.index(h)
        if Faction == True and 'faction' in h.lower():
            fact = headers.index(h)

    

    #Other is for nodes who might have degree 0 (titles or notes could get in there)
    other = []
    chapters = []
    ch = str()
    ch_number = 0
    for line in network:
        list_line = line.split('\t')
        
        #Gets chapter infomation (will be blank if there is none because of the "ch = str()" line above)
        if 'book' in list_line[0].lower() or 'scene' in list_line[0].lower() or 'chapter' in list_line[0].lower() or 'branch' in list_line[0].lower() or 'fragment' in list_line[0].lower() or 'prologue' in list_line[0].lower() or 'epliogue' in list_line[0].lower():
            ch = list_line[0].rstrip()
            ch_number += 1
            chapters += [ch]
            
        #The else is to not keep the lines with chapter names in them
        elif ch_start <= ch_number <= ch_end:        
            #This gets the node n_0 and tries to keep the accents
            book_name = list_line[0].strip()
            
            n_0 = get_normalised_name(book_name)
            
            if len(n_0) > 1:
                #UPDATE: Something strange is going on here with the comments I think
                #This adds each node to the Graph first and to other if it has no gender, friendly or hostile link
            
                if G.has_node(n_0) == False:# and Faction == False:
                    if 'book' in list_line[0].lower() or 'scene' in list_line[0].lower() or 'chapter' in list_line[0].lower() or 'branch' in list_line[0].lower() or 'fragment' in list_line[0].lower() or 'prologue' in list_line[0].lower() or 'epliogue' in list_line[0].lower():
                        pass
                    else:
                        G.add_node(n_0,sex=str(),page=str(),faction=str(),dead=str())
                    if len(list_line[friend].rstrip()) == 0 and len(list_line[hostile].rstrip()) == 0:# and len(list_line[gender]) == 0: 
                        other += [n_0]


                #We don't just want to add their attributes as they could have been added earlier so then
                # they will be replaced here, hence we just check if they are blank and then edit them if so
                if G.has_node(n_0) == True:
                    if len(G.nodes(data=True)[n_0]['sex']) == 0 and gender != -1:  
                        if list_line[gender].upper() == 'M':
                            G.nodes(data=True)[n_0]['sex'] = 'Male'
                        if list_line[gender].upper() == 'F':
                            G.nodes(data=True)[n_0]['sex'] = 'Female'
                        if list_line[gender].upper() not in ['M', 'F']:
                            G.nodes(data=True)[n_0]['sex'] = 'Unknown'
                        
                    if len(G.nodes(data=True)[n_0]['page']) == 0 and pg != -1:
                        G.nodes(data=True)[n_0]['page'] = list_line[pg]
                    
                    if len(G.nodes(data=True)[n_0]['dead']) == 0 and dead != -1:
                        G.nodes(data=True)[n_0]['dead'] = list_line[dead]
                        
                    if len(G.nodes(data=True)[n_0]['faction']) == 0 and Faction == True:
                        G.nodes(data=True)[n_0]['faction'] = list_line[fact]
                        
            begin = friend
            for i in range(begin,len(list_line)):
                
                #This gets the node n_i and tries to keep the accents
                friendnovelname = list_line[i].lstrip().rstrip()
                if len(friendnovelname) > 0:
                    n_i = get_normalised_name(friendnovelname)
                else:
                    n_i = ''
                                                       
                #Check this n_0 in G part
                if len(list_line[i]) > 0 and n_0 != n_i and len(n_i) > 1 and n_0 in G:
                    
                    #This adds the node n_i to to the network and gives them a blank gender, page no., etc.
                    if G.has_node(n_i) == False:
                        G.add_node(n_i,sex='Unknown',faction=str(),page=str(),dead=str())
                                     
                    #This is where the edges are added
                    if G.has_edge(n_0,n_i) == False and Hostile == True:
                        if i < hostile:                                
                            G.add_edge(n_0, n_i,weight=1,chap=ch,link='friendly')
                        if i >= hostile:
                            G.add_edge(n_0, n_i,weight=1,chap=ch,link='hostile')
                    if G.has_edge(n_0,n_i) == False and Hostile == False and i < hostile:
                        G.add_edge(n_0, n_i,weight=1,chap=ch) 

                    if G.has_edge(n_0,n_i) == True:
                        # This increases the weight if two characters are linked again but not if they are in the same chapter
                        if G.get_edge_data(n_0,n_i)['chap'] != ch: 
                            G.get_edge_data(n_0,n_i)['weight'] += 1
                        #This updates the type of link in case it changes as the story goes on (in case it changes from friendly to hostile for example)
                        if Hostile == True:
                            if G.get_edge_data(n_0,n_i)['link'] == 'friendly' and i >= hostile:
                                G.get_edge_data(n_0,n_i)['link'] == 'hostile'
                            if G.get_edge_data(n_0,n_i)['link'] == 'hostile' and i < hostile:
                                G.get_edge_data(n_0,n_i)['link'] == 'friendly'

            
    #Check to see if nodes in the list called "other" are in G, if so remove them
    #check = [i for i in other if i in G]
    #deg_0 = [i for i in check if G.degree(nbunch=i) == 0]
    #G.remove_nodes_from(deg_0)

    network.close()
    if chapter_list == False:
        return G
    if chapter_list == True:
        return [G,chapters]


#################################
#to build all:
if __name__ == "__main__":
    books = ['AGoT', 'ACoK', 'ASoS', 'AFFC', 'ADwD']
    G = nx.Graph()
    for book in books:
        G = build_network(f'in/novels/raw/{books.index(book) + 1}.{book}',G, ch_start=0)
    
