# -*- coding: cp1252 -*-
# -*- coding: utf-8 -*-
#
# Created by Pádraig Mac Carron

########################
#Import Libraries
import networkx as nx
########################

#Read network
#
# This reads a text file created by pasting the full Excel spreadsheet into a text file.
# It identifies the headers initially so it can determine a character's gender and what type
# of link it is (i.e. friendly or hostile).
#
# As it reads the document it looks for chapters (or books or scenes) and if two characters
# are linked in two different chapters it will increase the weight of that edge. 
#
# Chapter information is also added to the edge.
# If the argument "chapter_list=True" is added when the function is called
# it will output a list of chapters in the order they appear also as networkx
# will not keep the order.
#
# Some datasets have a faction as well, if that is to be 
# included then add the argument "Faction=True".
#
# Set "Hostile=False" if there are no hostile edges
#

def build_network(network,start_net=None, ch_start=0, ch_end=None, chapter_list=False,Faction=False,Hostile=True,file_type=".tsv"):
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
            n_0 = list_line[0].strip()
            
            if len(n_0) > 1:
                #UPDATE: Something strange is going on here with the comments I think
                #This adds each node to the Graph first and to other if it has no gender, friendly or hostile link
            
                if G.has_node(n_0) == False:# and Faction == False:
                    G.add_node(n_0,Sex=str(),page=str(),faction=str(),dead=str())
                    if len(list_line[friend].rstrip()) == 0 and len(list_line[hostile].rstrip()) == 0:# and len(list_line[gender]) == 0: 
                        other += [n_0]

                #We don't just want to add their attributes as they could have been added earlier so then
                # they will be replaced here, hence we just check if they are blank and then edit them if so
                if G.has_node(n_0) == True:
                    if len(G.nodes(data=True)[n_0]['Sex']) == 0 and gender != -1:     
                        G.nodes(data=True)[n_0]['Sex'] = list_line[gender].upper()
                        
                    if len(G.nodes(data=True)[n_0]['page']) == 0 and pg != -1:
                        G.nodes(data=True)[n_0]['page'] = list_line[pg]
                    
                    if len(G.nodes(data=True)[n_0]['dead']) == 0 and dead != -1:
                        G.nodes(data=True)[n_0]['dead'] = list_line[dead]
                        
                    if len(G.nodes(data=True)[n_0]['faction']) == 0 and Faction == True:
                        G.nodes(data=True)[n_0]['faction'] = list_line[fact]
                        
            begin = friend
            for i in range(begin,len(list_line)):
                
                #This gets the node n_i and tries to keep the accents
                n_i = list_line[i].lstrip().rstrip()
                                                                            #Check this n_0 in G part
                if len(list_line[i]) > 0 and n_0 != n_i and len(n_i) > 1 and n_0 in G:
                    
                    #This adds the node n_i to to the network and gives them a blank gender, page no., etc.
                    if G.has_node(n_i) == False:
                        G.add_node(n_i,Sex=str(),faction=str(),page=str(),dead=str())
                                     
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
# books = ['AGoT', 'ACoK', 'ASoS', 'AFFC', 'ADwD']
# G = nx.Graph()
# for book in books:
#     G = build_network(G, start_net=G, ch_start=0)
    
