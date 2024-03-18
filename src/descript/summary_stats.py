# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:28:54 2023

@author: shane
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:33:52 2023

@author: shane
"""

#Libraries
import numpy as np
import networkx as nx
import pandas as pd
from modularity import modularity



def summary_stats(networks: list, medium: str, saveloc: str):
    """
    

    Parameters
    ----------
    networks : list
        list of networks (ensure in correct order!! E.g. [ch_1, ch_2,...]).
    medium : str
        comic/show/book
    saveloc : str
        path to save csv of results
    Returns
    -------
    Data_Table : pd.DataFrame()

    """
    Data_Table = pd.DataFrame()
    #Reading networks
    graphs = []
    for network in networks:
        #Get the network name
    
        G = nx.read_graphml(open(network,'rb'))
        
        graphs.append(G)
    
    #Stats for size of network
    
    for i,g in enumerate(graphs):
        G = g

        
        medium = medium
        if medium == 'book':
            Name = network.split('/')[-1].split('.')[1]
        if medium == 'comic':
            if len(networks) < 1000:
                num = networks[1].split('/')[-1].split('.')[0]
                Name = f'chapter_{num}'
            else:
                num = networks[1].split('/')[-1].split('.')[0]
                Name = f'scene_{num}'
                
        if medium == 'show':
            num = networks[1].split('/')[-1].split('.')[0]
            if len(networks) < 1000:
                Name = f'episode_{num}'
            else:
                Name = f'scene_{num}'
            
        N, M = len(G), G.size(weight='weight')

    
        try:
            GC = G.subgraph(max(nx.connected_components(G), key =len))
        except ValueError:
            GC = 'NaN'
        try:
            GC_size = round((len(GC)/N)*100,3)
        except ZeroDivisionError:
            GC_size = 'NaN'
    
        
        #Degree info#
        deg = dict(G.degree(weight='weight'))
        degree = list(deg.values())
    
        d = np.array(degree)
        
        try:
            average_degree = round(float(sum(degree))/N,3)
        except ZeroDivisionError:
            average_degree = 'NaN'
            
        if len(d) != 0:
            Max_degree = d.max()
        else:
            Max_degree = 0
        ##Male-Female##
 
        
        men = len([d.get('sex', 'Unknown') for (u,d) in G.nodes(True) if d.get('sex', 'Unknown') == 'Male'])
        women = len([d.get('sex', 'Unknown') for (u,d) in G.nodes(True) if d.get('sex', 'Unknown') == 'Female'])
        Number_Males = men
        Number_females = women
        if medium == 'comic':
            mix = len([d.get('sex', 'Unknown') for (u,d) in G.nodes(True) if d.get('sex', 'Unknown') == 'Mixed'])
            Number_Mixed = mix
            Number_Unknown = len(G) - (women + men + mix)
        else:
            Number_Unknown = len(G) - (women+men)
        
        if len(G) != 0:
            Percentage_Men = round((Number_Males/len(G))*100,3)
            Percentage_women = round((women/len(G))*100,3)
            Percentage_Unknown = round((Number_Unknown/len(G))*100,3)
            if medium == 'comic':
                Percentage_Mix = round((Number_Mixed/len(G))*100,3)
        else:
            Percentage_women, Percentage_Unknown = 'NaN', 'NaN'
    
        ##Male-Female Interactions
    
        mm = 0
        ff = 0
        mf = 0
        mu = 0
        fu = 0
        uu = 0
        mixf = 0
        mixu = 0
        mixm = 0
        mix2 = 0
        
        for u,v in G.edges():
            if len(G.nodes()[u]) != 0 and len(G.nodes()[v]) != 0 :
                if G.nodes()[u].get('sex', 'Unknown') == 'Male' and G.nodes()[v].get('sex', 'Unknown') == 'Male':
                    mm += G.edges()[u,v]['weight']
                if G.nodes()[u].get('sex', 'Unknown') == 'Female' and G.nodes()[v].get('sex', 'Unknown') == 'Female':
                    ff += G.edges()[u,v]['weight']
                if G.nodes()[u].get('sex', 'Unknown') == 'Male' and G.nodes()[v].get('sex', 'Unknown') == 'Female' or G.nodes()[u].get('sex', 'Unknown') == 'Female' and G.nodes()[v].get('sex', 'Unknown') == 'Male':
                    mf += G.edges()[u,v]['weight']
                if G.nodes()[u].get('sex', 'Unknown') == 'Male' and G.nodes()[v].get('sex', 'Unknown') == 'Unknown' or G.nodes()[u].get('sex', 'Unknown') == 'Unknown' and G.nodes()[v].get('sex', 'Unknown') == 'Male':
                    mu += G.edges()[u,v]['weight']
                if G.nodes()[u].get('sex', 'Unknown') == 'Female' and G.nodes()[v].get('sex', 'Unknown') == 'Unknown' or G.nodes()[u].get('sex', 'Unknown') == 'Unknown' and G.nodes()[v].get('sex', 'Unknown') == 'Female':
                    fu += G.edges()[u,v]['weight']
                if G.nodes()[u].get('sex', 'Unknown') == 'Unknown' and G.nodes()[v].get('sex', 'Unknown') == 'Unknown':
                    uu += G.edges()[u,v]['weight']
                if medium == 'comic':
                    if G.nodes()[u].get('sex', 'Unknown') == 'Mixed' and G.nodes()[v].get('sex', 'Unknown') == 'Female' or G.nodes()[u].get('sex', 'Unknown') == 'Female' and G.nodes()[v].get('sex', 'Unknown') == 'Mixed':
                        mixf += G.edges()[u,v]['weight']
                    if G.nodes()[u].get('sex', 'Unknown') == 'Mixed' and G.nodes()[v].get('sex', 'Unknown') == 'Unknown' or G.nodes()[u].get('sex', 'Unknown') == 'Unknown' and G.nodes()[v].get('sex', 'Unknown') == 'Mixed':
                        mixu += G.edges()[u,v]['weight']
                    if G.nodes()[u].get('sex', 'Unknown') == 'Mixed' and G.nodes()[v].get('sex', 'Unknown') == 'Male' or G.nodes()[u].get('sex', 'Unknown') == 'Male' and G.nodes()[v].get('sex', 'Unknown') == 'Mixed':
                        mixm += G.edges()[u,v]['weight']
                    if G.nodes()[u].get('sex', 'Unknown') == 'Mixed' and G.nodes()[v].get('sex', 'Unknown') == 'Mixed':
                        mix2 += G.edges()[u,v]['weight']
            
                    
        if M == 0:
            FF, PercentFF = ff, 'NaN'
            MF, PercentMF = mf, 'NaN'
            FU, PercentFU = fu, 'NaN'
            MU, PercentMU = mu, 'NaN'
            MM, PercentMM = mm, 'NaN'
            UU, PercentUU = uu, 'NaN'

        else:
            FF, PercentFF = ff, round((ff/M)*100,3)
            MF, PercentMF = mf, round((mf/M)*100,3)
            FU, PercentFU = fu, round((fu/M)*100,3)
            MU, PercentMU = mu, round((mu/M)*100,3)
            MM, PercentMM = mm, round((mm/M)*100,3)
            UU, PercentUU = uu, round((uu/M)*100,3)
                
        
    
    
        if medium == 'comic':
            if M == 0:
                MIXM, PercentMIXM = mixm, 'NaN'
                MIXF, PercentMIXF = mixf, 'NaN'
                MIXU, PercentMIXU = mixu, 'NaN'
                MIX2, PercentMIX2 = mix2, 'NaN'
            else:
                MIXM, PercentMIXM = mixm, round((mixm/M)*100,3)
                MIXF, PercentMIXF = mixf, round((mixf/M)*100,3)
                MIXU, PercentMIXU = mixu, round((mixu/M)*100,3)
                MIX2, PercentMIX2 = mix2, round((mix2/M)*100,3)

        female_M_No_U =  mf + ff
        female_M_G = mf+ff+fu
        male_M = mm+mu+mf
        try:
            frac_F_No_U =  round((mf + ff)/(mf+ff+mm),3)
        except ZeroDivisionError:
            frac_F_No_U = 'NaN'
        try:
            frac_F_G = round(((mf+ff+fu)/M)*100,3)
        except ZeroDivisionError:
            frac_F_G = 'NaN'
        try:
            frac_M_G = round(((mm+mu+mf)/M)*100,3)
        except ZeroDivisionError:
            frac_M_G = 'NaN'
        ##Important Nodes by degree
    
        Imp_M = 0
        Imp_F = 0
        Imp_U = 0

        for u in G.nodes:
            if G.nodes()[u].get('sex', 'Unknown') == 'Male':
                if G.degree[u] > average_degree:
                    Imp_M += 1
            if G.nodes()[u].get('sex', 'Unknown') == 'Female':
                if G.degree[u] > average_degree:
                    Imp_F += 1
            if G.nodes()[u].get('sex', 'Unknown') == 'Unknown':
                if G.degree[u] > average_degree:
                    Imp_U += 1
                    
        if medium == 'comic':
            Imp_Mix = 0

            for u in G.nodes:
                if G.nodes()[u].get('sex', 'Unknown') == 'Mixed':
                    if G.degree[u] > average_degree:
                        Imp_Mix += 1
                        
        if N != 0:
            Percent_Imp_U = round((Imp_U/N)*100,3)
        else:
            Percent_Imp_U = 'NaN'
    
           
        ## Betweenness Centrality ##
        BCF = 0
        BCU = 0
        BCM = 0
        BCMix = 0
        bet = nx.betweenness_centrality(G, weight='weight')
        b_sort = sorted(bet, key=bet.get,reverse=True)
        for i,u in enumerate(b_sort):
            if len(G.nodes()[u].get('sex', 'Unknown')) > 0 and G.nodes()[u].get('sex', 'Unknown') == 'Female':
                BCF += 1
            if i > (len(G))/10:
                break
        for i,u in enumerate(b_sort):
            if len(G.nodes()[u].get('sex', 'Unknown')) > 0 and G.nodes()[u].get('sex', 'Unknown') == 'Male':
                BCM += 1
            if i > (len(G))/10:
                break
        for i,u in enumerate(b_sort):
            if len(G.nodes()[u].get('sex', 'Unknown')) > 0 and G.nodes()[u].get('sex', 'Unknown') == 'Unknown':
                BCU += 1
            if i > (len(G))/10:
                break
        if medium == 'comic':
            for i,u in enumerate(b_sort):
                if len(G.nodes()[u].get('sex', 'Unknown')) > 0 and G.nodes()[u].get('sex', 'Unknown') == 'Mixed':
                    BCMix += 1
                if i > (len(G))/10:
                    break
    
       ### BC bottom
        UBCF = 0
        UBCU = 0
        UBCM = 0
        UBCMix = 0
        ub_sort = sorted(bet, key=bet.get, reverse= False)
        for i,u in enumerate(ub_sort):
            if len(G.nodes()[u].get('sex', 'Unknown')) > 0 and G.nodes()[u].get('sex', 'Unknown') == 'Female':
                UBCF += 1
            if i >= (len(G))/10:
                break
        for i,u in enumerate(ub_sort):
            if len(G.nodes()[u].get('sex', 'Unknown')) > 0 and G.nodes()[u].get('sex', 'Unknown') == 'Male':
                UBCM += 1
            if i >= (len(G))/10:
                break
        for i,u in enumerate(ub_sort):
            if len(G.nodes()[u].get('sex', 'Unknown')) > 0 and G.nodes()[u].get('sex', 'Unknown') == 'Unknown':
                UBCU += 1
            if i >= (len(G))/10:
                break
        if medium == 'comic':
            for i,u in enumerate(ub_sort):
                if len(G.nodes()[u].get('sex', 'Unknown')) > 0 and G.nodes()[u].get('sex', 'Unknown') == 'Mixed':
                    UBCMix += 1
                if i >= (len(G))/10:
                    break
        
        MF_10 = round((men+women)/10,3)
        G_10 = round(len(G)/10,3)
        if G_10 == 0:
           P_BCF = 0
           P_BCU = 0
           P_BCM = 0
           P_UBCF = 0
           P_UBCU = 0
           P_UBCM = 0
        else:
            P_BCF = round((BCF/(G_10))*100,3)
            P_BCU = round((BCU/(G_10))*100,3)
            P_BCM = round((BCM/(G_10))*100,3)
            P_UBCF = round((UBCF/(G_10))*100,3)
            P_UBCU = round((UBCU/(G_10))*100,3)
            P_UBCM = round((UBCM/(G_10))*100,3)
              
        if Number_females == 0:
            P_BCF_of_F = 0
        else:
            P_BCF_of_F = round((BCF/Number_females)*100,3)    
    
        #Average Path Length#
        apl = []
        diam = []
        len_comp = []
        def connected_component_subgraphs(G):
                for c in nx.connected_components(G):
                        yield G.subgraph(c)
    
        for g in connected_component_subgraphs(G):
                if len(g) > 1:
                        apl += [nx.average_shortest_path_length(g)]
                        diam += [nx.diameter(g)]
                        len_comp += [len(g)]
    
        d = 0
        for i in range(len(apl)):
                d += apl[i]*len_comp[i]*(len_comp[i]-1)
    
        n_m = 0
        for i in len_comp:
                n_m += i*(i-1.)
    
        try:
            diameter = max(diam)
        except ValueError:
            diameter = 'NaN'
            
                
        try:
            path_length = round(d/n_m,3)
        except ZeroDivisionError:
            path_length = 'NaN'
        
        if average_degree in [0,'NaN'] or N == 0:
            Random_path_length = 'NaN'
        else:
            Random_path_length = round(np.log(N)/np.log(average_degree),3)
        #Clustering#
        Clustering = nx.clustering(G, weight='weight')
        Clus1 = Clustering.values()
    
        if N == 0:
            CC = 'NaN'
        else:
            CC = round(sum(Clus1)/N,3)
    
        if average_degree == 'NaN' or N in [0, 'NaN']:
            Random_Clustering = 'NaN'
        else:
            Random_Clustering = round(average_degree/N,3)
    
        #Transitivity#
        Transitivity = round(nx.transitivity(G),3)
    
        #Assortativity#
        try:
            Assortativity = round(nx.degree_assortativity_coefficient(G, weight='weight'),3)
        except ValueError:
            Assortativity = 'NaN'
        #Density
        Density = round(nx.density(G),3)
    
        ##Average betweenness centrality of network
        bet_val = list(bet.values())
        B = np.array(bet_val)
    
        if N != 0:
            BC_Average = float(sum(B))/N
        else:
            BC_Average = 'NaN'
        ## Categorical assortativity
        try:
            Cat_S = modularity(G,'sex')
        except ZeroDivisionError:
            Cat_S = 'NaN'
                
    
        Stats_Gender = {'Name':Name,'Medium':medium,'N':N,'M':M,'<k>':average_degree, 'Max_degree':Max_degree,'Diameter':diameter, 'Average path length':path_length, 'Random path length':Random_path_length, 'Clustering':CC,
                        'Random Clustering': Random_Clustering, 'Transitivity':Transitivity, 'Assortativity':Assortativity, 'Density':Density, 'GC Size (%)':GC_size, '% Women':Percentage_women, '# Women':Number_females, 'Fraction female interaction': frac_F_No_U,
                        '% Unknown':Percentage_Unknown, '10% M&F Characters':MF_10, '# of M nodes with k><k>':Imp_M, '# of F nodes with k><k>':Imp_F,
                        '# of U nodes with k><k>':Imp_U, '% of U nodes with k><k>':Percent_Imp_U, '10% of all nodes':G_10, '# of Unknowns in top 10%(BC)':BCU, '% of Unknowns in top 10%(BC)':P_BCU, '% F-F Interactions': PercentFF,
                         '# of F-F Interactions':FF, '% F-M Interactions':PercentMF, '# of F-M Interactions':MF, '% F-Any Interactions':frac_F_G, '# of F-Any Interactions': female_M_G, '% M-Any Interactions': frac_M_G, '# of M-Any Interactions': male_M, '% M-M Interactions':PercentMM,
                        '# of M-M Interactions':MM, 'Average Betweenness':BC_Average, '# of men in top 10%(BC)':BCM, '% of men in top 10%(BC)':P_BCM, '# of men': men, '# of men in top 10%(BC)':BCM, '% of men in bottom 10%(BC)':P_UBCM,'# of men in bottom 10%(BC)':UBCM ,
                        '% of women in bottom 10%(BC)':P_UBCF,'# of women in bottom 10%(BC)':UBCF, '% of unknowns in bottom 10%(BC)':P_UBCU,'# of unknowns in bottom 10%(BC)':UBCU, '# of women in top 10%(BC)':BCF, '% of women in top 10%(BC)':P_BCF, 
                        '% of Important women as % of all women':P_BCF_of_F,'Categorical assortativity_Sex': Cat_S}
    
        if medium == 'comic':
            mixed = { '% of Mix-M':PercentMIXM, '% of Mix-F': PercentMIXF, '% of Mix-U':PercentMIXU, '% of Mix-Mix': PercentMIX2, '% Men':Percentage_Men, '% mix':Percentage_Mix, '# of Mix nodes with k><k>':Imp_Mix, '# of mixed in top 10%(BC)':BCMix, '# of mixed in bottom 10%(BC)':UBCMix}
            Stats_Gender.update(mixed)
        Data_Table = Data_Table.append(Stats_Gender, ignore_index=True)
        Data_Table = Data_Table[['Name','Medium', 'N', 'M', '<k>', 'Max_degree', 'Diameter', 'Average path length', 'Random path length', 'Clustering', 'Random Clustering', 'Transitivity', 'Assortativity', 'Density', 'GC Size (%)', '% Women', '# Women', 'Fraction female interaction',
                        '% Unknown', '10% M&F Characters', '# of M nodes with k><k>', '# of F nodes with k><k>',
                        '# of U nodes with k><k>', '% of U nodes with k><k>', '10% of all nodes', '# of Unknowns in top 10%(BC)', '% of Unknowns in top 10%(BC)', '% F-F Interactions',
                        '# of F-F Interactions', '% F-M Interactions', '# of F-M Interactions', '% F-Any Interactions', '# of F-Any Interactions', '% M-Any Interactions', '# of M-Any Interactions', '% M-M Interactions',
                        '# of M-M Interactions', 'Average Betweenness', '# of men in top 10%(BC)', '% of men in top 10%(BC)','# of men', '% of men in bottom 10%(BC)','# of men in bottom 10%(BC)', '% of women in bottom 10%(BC)',
                        '# of women in bottom 10%(BC)', '% of unknowns in bottom 10%(BC)','# of unknowns in bottom 10%(BC)', '# of women in top 10%(BC)', '% of women in top 10%(BC)', '% of Important women as % of all women', 
                        'Categorical assortativity_Sex']]

      
        
        Data_Table.to_csv(saveloc, index=False)
    return Data_Table