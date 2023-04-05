#Libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from modularity import modularity
import os

dir_path = r'C:\Users\mads3\OneDrive - Coventry University\GoT networks\C_IS'

##########
#Datasets
networks = [ ]

for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path,path)):
        networks.append(f'C_IS/{path}')
#print(networks)

#data = {'Titles':['Narrative', 'N', 'M', '<k>', 'Max_degree', 'Average path length', 'Clustering', 'Transitivity', 'Assortativity', 'Density', 'GC Size (%)']}
Data_Table_Standard= pd.DataFrame()
Data_Table_Gender=pd.DataFrame()
pd.set_option('display.max_colwidth',40)
#Reading networks
graphs = {}
for network in networks:
    #Get the network name
    net = network.split('/')[-1].split('.')[0]

#A Song of Ice and Fire has been pre-saved
    if network == 'Njal Manual.tsv':

        #Create empty graph
        G = nx.Graph()

        #Reads through text files and creates networks
        with open(network,'rb') as f:
            headers = f.readline().decode('cp1252').strip().split('\t')
            # Change order of headers for Friendly
            links, h_link = headers.index('Links'), headers.index('Other Links')
            gend = headers.index('Gender')
            for line in f:
                l = line.decode('cp1252').split('\t')
                if line[0] == b'%':
                    continue
                l = [u.strip().replace('"','') for u in l]
                u = l[0]
                if len(u) > 1:
                    if u in G and len(G.nodes()[u]) < 1:
                        G.nodes()[u]['Sex'] = l[gend].upper()
                    G.add_node(u,Sex=l[gend].upper())
                for v in l[links:]:
                    ltype = 'Friendly'
                    if l.index(v) >= h_link:
                        ltype = 'Hostile'
                        #Uncomment this next line to remove hostile links
#                        continue
                    if len(u) > 1 and len(v) > 1:
                        if G.has_edge(u,v):
                            G.get_edge_data(u,v)['weight'] += 1
                        else:
                            G.add_edge(u,v,weight=1, link=ltype)


        for u in [v for v in G if len(G.nodes()[v]) < 1]:
            G.nodes()[u]['Sex'] = ''
            
    #Getting gml data
    if network != 'Njal Manual.tsv':
        #G = nx.read_gml(open(network,'rb'))
        G = nx.read_graphml(open(network,'rb'))
         


    #    import pickle
    #    G = pickle.load(open(network,'rb'))

    #G.remove_nodes_from([u for u in G if G.nodes()[u]['Sex'] == 'Female']) 
    #G.remove_nodes_from([u for u in G if G.degree(u)==0])
    
    graphs[net] = G


#Stats for size of network

for i,g in enumerate(graphs):
    print(i)
    G = graphs[g]
    #print("\n---------\n",g)
    Name = g
    N, M = len(G), len(G.edges())
    #Data_Table = Data_Table.append({'Narrative':g,'N':N, 'M':M},ignore_index=True)

 #   print(G.edges())

    GC = G.subgraph(max(nx.connected_components(G), key =len))
    #print(len(GC))
    GC_size = round((len(GC)/N)*100,3)

    ##Get node attribute

#    print(nx.get_node_attributes(G, 'Sex'))

    
    #Degree info#
    deg = dict(G.degree())
    degree = list(deg.values())

    d = np.array(degree)
    #print(d)

    average_degree = round(float(sum(degree))/N,3)

    Max_degree = d.max()

    ##Male-Female##

    if g == 'Iliad':
        gend = {}
        with open('data/iliad_gender combined.csv') as f:
            for linr in f:
                l = line.strip().split(',')
                if len(l) > 1:
                    gend[l[0].strip()] = l[1].upper()
        for u in gend:
            if u in G and len(G.nodes()[u]['Sex'])==0:
                G.nodes()[u]['Sex'] = gend[u]
 
    men = len([d['Sex'] for (u,d) in G.nodes(True) if d['Sex'] == 'Male'])
    women = len([d['Sex'] for (u,d) in G.nodes(True) if d['Sex'] == 'Female'])
    mix = len([d['Sex'] for (u,d) in G.nodes(True) if d['Sex'] == 'Mixed'])
    unknown = len([d['Sex'] for (u,d) in G.nodes(True) if d['Sex'] == 'Unknown'])
    Number_Males = men
    Number_females = women
    Number_Unknown =  unknown  #len(G) - (women+men)
    Number_Mixed = mix
    
    Percentage_women = round((women/len(G))*100,3)
    Percentage_Unknown = round((Number_Unknown/len(G))*100,3)
    Percentage_Men = round((Number_Males/len(G))*100,3)
    Percentage_Mix = round((Number_Mixed/len(G))*100,3)

    

    ##Male-Female Interactions

    mm = 0
    ff = 0
    mf = 0
    mu = 0
    fu = 0
    uu = 0
    mixm = 0
    mixf = 0
    mixu =0
    mix2 =0
    
    for u,v in G.edges():
        if len(G.nodes()[u]) != 0 and len(G.nodes()[v]) != 0 :
            if G.nodes()[u]['Sex'] == 'Male' and G.nodes()[v]['Sex'] == 'Male':
                mm += 1
            if G.nodes()[u]['Sex'] == 'Female' and G.nodes()[v]['Sex'] == 'Female':
                ff += 1
            if G.nodes()[u]['Sex'] == 'Male' and G.nodes()[v]['Sex'] == 'Female' or G.nodes()[u]['Sex'] == 'Female' and G.nodes()[v]['Sex'] == 'Male':
                mf += 1
            if G.nodes()[u]['Sex'] == 'Male' and G.nodes()[v]['Sex'] == 'Unknown' or G.nodes()[u]['Sex'] == '' and G.nodes()[v]['Sex'] == 'Male':
                mu += 1
            if G.nodes()[u]['Sex'] == 'Female' and G.nodes()[v]['Sex'] == 'Unknown' or G.nodes()[u]['Sex'] == '' and G.nodes()[v]['Sex'] == 'Female':
                fu += 1
            if G.nodes()[u]['Sex'] == 'Unknown' and G.nodes()[v]['Sex'] == 'Unknown':
                uu += 1
            if G.nodes()[u]['Sex'] == 'Mixed' and G.nodes()[v]['Sex'] == 'Female' or G.nodes()[u]['Sex'] == 'Female' and G.nodes()[v]['Sex'] == 'Mixed':
                mixf += 1
            if G.nodes()[u]['Sex'] == 'Mixed' and G.nodes()[v]['Sex'] == 'Unknown' or G.nodes()[u]['Sex'] == 'Unknown' and G.nodes()[v]['Sex'] == 'Mixed':
                mixu += 1
            if G.nodes()[u]['Sex'] == 'Mixed' and G.nodes()[v]['Sex'] == 'Male' or G.nodes()[u]['Sex'] == 'Male' and G.nodes()[v]['Sex'] == 'Mixed':
                mixm += 1
            if G.nodes()[u]['Sex'] == 'Mixed' and G.nodes()[v]['Sex'] == 'Mixed':
                mix2 += 1
            
            

    if M == 0:
        FF, PercentFF = ff, 'no edges - divide by 0 error'
        MF, PercentMF = mf, 'no edges - divide by 0 error'
        FU, PercentFU = fu, 'no edges - divide by 0 error'
        MU, PercentMU = mu, 'no edges - divide by 0 error'
        MM, PercentMM = mm, 'no edges - divide by 0 error'
        UU, PercentUU = uu, 'no edges - divide by 0 error'
        MIXM, PercentMIXM = mixm, 'no edges - divide by 0 error'
        MIXF, PercentMIXF = mixf, 'no edges - divide by 0 error'
        MIXU, PercentMIXU = mixu, 'no edges - divide by 0 error'
        MIX2, PercentMIX2 = mix2, 'no edges - divide by 0 error'
    else:
        FF, PercentFF = ff, round((ff/M)*100,3)
        MF, PercentMF = mf, round((mf/M)*100,3)
        FU, PercentFU = fu, round((fu/M)*100,3)
        MU, PercentMU = mu, round((mu/M)*100,3)
        MM, PercentMM = mm, round((mm/M)*100,3)
        UU, PercentUU = uu, round((uu/M)*100,3)
        MIXM, PercentMIXM = mixm, round((mixm/M)*100,3)
        MIXF, PercentMIXF = mixf, round((mixf/M)*100,3)
        MIXU, PercentMIXU = mixu, round((mixu/M)*100,3)
        MIX2, PercentMIX2 = mix2, round((mix2/M)*100,3)

    female_M_No_U =  mf + ff
    female_M_G = mf+ff+fu
    male_M = mm+mu+mf

    if M == 0:
        frac_F_No_U =  'no edges - divide by 0 error'
        frac_F_G = 'no edges - divide by 0 error'
        frac_M_G = 'no edges - divide by 0 error'
    else:
        frac_F_No_U =  round((mf + ff)/(mf+ff+mm),3)
        frac_F_G = round(((mf+ff+fu)/M)*100,3)
        frac_M_G = round(((mm+mu+mf)/M)*100,3)

    ##Important Nodes by degree

    Imp_M = 0
    Imp_F = 0
    Imp_U = 0
    Imp_Mix = 0
    for u in G.nodes:
        if G.nodes()[u]['Sex'] == 'Male':
            if G.degree[u] > average_degree:
                Imp_M += 1
        if G.nodes()[u]['Sex'] == 'Female':
            if G.degree[u] > average_degree:
                Imp_F += 1
        if G.nodes()[u]['Sex'] == 'Unknown':
            if G.degree[u] > average_degree:
                Imp_U += 1
        if G.nodes()[u]['Sex'] == 'Mixed':
            if G.degree[u] > average_degree:
                Imp_Mix += 1
                
    Percent_Imp_U = round((Imp_U/N)*100,3)

       
    ## Betweenness Centrality ##
    BCF = 0
    BCU = 0
    BCM = 0
    BCMix = 0
    bet = nx.betweenness_centrality(G, weight='weight')
    b_sort = sorted(bet, key=bet.get,reverse=True)
    for i,u in enumerate(b_sort):
        if len(G.nodes()[u]['Sex']) > 0 and G.nodes()[u]['Sex'] == 'Female':
            BCF += 1
        if i > (len(G))/10:
            break
    for i,u in enumerate(b_sort):
        if len(G.nodes()[u]['Sex']) > 0 and G.nodes()[u]['Sex'] == 'Male':
            BCM += 1
        if i > (len(G))/10:
            break
    for i,u in enumerate(b_sort):
        if len(G.nodes()[u]['Sex']) > 0 and G.nodes()[u]['Sex'] == 'Unknown':
            BCU += 1
        if i > (len(G))/10:
            break
    for i,u in enumerate(b_sort):
        if len(G.nodes()[u]['Sex']) > 0 and G.nodes()[u]['Sex'] == 'Mixed':
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
        if len(G.nodes()[u]['Sex']) > 0 and G.nodes()[u]['Sex'] == 'Female':
            UBCF += 1
        if i >= (len(G))/10:
            break
    for i,u in enumerate(ub_sort):
        if len(G.nodes()[u]['Sex']) > 0 and G.nodes()[u]['Sex'] == 'Male':
            UBCM += 1
        if i >= (len(G))/10:
            break
    for i,u in enumerate(ub_sort):
        if len(G.nodes()[u]['Sex']) > 0 and G.nodes()[u]['Sex'] == 'Unknown':
            UBCU += 1
        if i >= (len(G))/10:
            break
    for i,u in enumerate(ub_sort):
        if len(G.nodes()[u]['Sex']) > 0 and G.nodes()[u]['Sex'] == 'Mixed':
            UBCMix += 1
        if i >= (len(G))/10:
            break
    
    MF_10 = round((men+women)/10,3)
    G_10 = round(len(G)/10,3)
    P_BCF = round((BCF/(G_10))*100,3)
    #P_BCU = 0
    P_BCU = round((BCU/(G_10))*100,3)
    P_BCM = round((BCM/(G_10))*100,3)
    if Number_females == 0:
        P_BCF_of_F = 0
    else:
        P_BCF_of_F = round((BCF/Number_females)*100,3)

    P_UBCF = round((UBCF/(G_10))*100,3)
    #P_BCU = 0
    P_UBCU = round((UBCU/(G_10))*100,3)
    P_UBCM = round((UBCM/(G_10))*100,3)
      
    

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

    if M == 0:
        diameter = 'no edges - divide by 0 error'
        path_length = 'no edges - divide by 0 error'
    else:
        diameter = max(diam)
        path_length = round(d/n_m,3)

    

    Random_path_length = round(np.log(N)/np.log(average_degree),3)
    #Clustering#
    Clustering = nx.clustering(G)
    Clus1 = Clustering.values()

    CC = round(sum(Clus1)/N,3)

    Random_Clustering = round(average_degree/N,3)

    #Transitivity#
    Transitivity = round(nx.transitivity(G),3)

    #Assortativity#
    Assortativity = round(nx.degree_assortativity_coefficient(G),3)

    #Density
    OG_Density = nx.density(G)
    #print(OG_Density)
    Density = round(nx.density(G),3)

    ##Average betweenness centrality of network
    bet_val = list(bet.values())
    B = np.array(bet_val)

    BC_Average = float(sum(B))/N

    ## Categorical assortativity
    try:
        Cat_S = modularity(G,'Sex')
    except ZeroDivisionError:
        Cat_S = 'error'

    ##assign house if unknown
    
    h = 0
    for n in G.nodes():
        try:
            G.nodes()[n]['house']
        except KeyError:
            h += 1
            G.nodes()[n]['house'] = f'unknown_{h}'


    if M == 0:
        Cat_H = 'no edges - divide by 0 error'
    else:
        try:
            Cat_H = modularity(G,'house')
        except ZeroDivisionError:
            Cat_H = 'error'


##### Removed some of the stats that aren't used as much e.g. # of each edge type (M-F, F-F...) but kept percentages.
    



    Stats_Gender = {'Narrative':Name,'N':N,'M':M,'<k>':average_degree, 'Max_degree':Max_degree, 'Average path length':path_length, 'Random path length':Random_path_length, 'Clustering':CC,
                    'Random Clustering': Random_Clustering, 'Transitivity':Transitivity, 'Assortativity':Assortativity, 'Density':Density, 'GC Size (%)':GC_size, '% Women':Percentage_women, '# Women':Number_females, 'Fraction female interaction': frac_F_No_U,
                    '% Unknown':Percentage_Unknown, '10% M&F Characters':MF_10, '# of M nodes with k><k>':Imp_M, '# of F nodes with k><k>':Imp_F,
                    '# of U nodes with k><k>':Imp_U, '% of U nodes with k><k>':Percent_Imp_U, '10% of all nodes':G_10, '# of Unknowns in top 10%(BC)':BCU, '% of Unknowns in top 10%(BC)':P_BCU, '% F-F Interactions': PercentFF,
                     '% F-M Interactions':PercentMF, '% F-Any Interactions':frac_F_G, '% M-Any Interactions': frac_M_G, '% M-M Interactions':PercentMM,
                     'Average Betweenness':BC_Average, '# of men in top 10%(BC)':BCM, '# of men': men, '# of men in bottom 10%(BC)':UBCM ,
                    '# of women in bottom 10%(BC)':UBCF, '# of unknowns in bottom 10%(BC)':UBCU, '# of women in top 10%(BC)':BCF, 'Categorical assortativity_Sex': Cat_S, 'Categorical assortativity_House': Cat_H, '% of Mix-M':PercentMIXM, '% of Mix-F': PercentMIXF, '% of Mix-U':PercentMIXU, '% of Mix-Mix': PercentMIX2, '% Men':Percentage_Men, '% mix':Percentage_Mix, '# of Mix nodes with k><k>':Imp_Mix, '# of mixed in top 10%(BC)':BCMix, '# of mixed in bottom 10%(BC)':UBCMix}

    Data_Table_Gender = Data_Table_Gender.append(Stats_Gender, ignore_index=True)
    Data_Table_Gender = Data_Table_Gender[['Narrative', 'N', 'M', '<k>', 'Max_degree', 'Average path length', 'Random path length', 'Clustering', 'Random Clustering', 'Transitivity', 'Assortativity', 'Density', 'GC Size (%)', '% Women', '# Women', 'Fraction female interaction',
                    '% Unknown', '10% M&F Characters', '# of M nodes with k><k>', '# of F nodes with k><k>',
                    '# of U nodes with k><k>', '% of U nodes with k><k>', '10% of all nodes', '# of Unknowns in top 10%(BC)', '% of Unknowns in top 10%(BC)', '% F-F Interactions',
                     '% F-M Interactions', '% F-Any Interactions', '% M-Any Interactions', '% M-M Interactions',
                     'Average Betweenness', '# of men in top 10%(BC)', '# of men', '# of men in bottom 10%(BC)', '# of women in bottom 10%(BC)','# of unknowns in bottom 10%(BC)', '# of women in top 10%(BC)',  'Categorical assortativity_Sex','Categorical assortativity_House', '% of Mix-M', '% of Mix-F', '% of Mix-U', '% of Mix-Mix',
                    '% of Mix-M', '% of Mix-F', '% of Mix-U', '% of Mix-Mix', '% Men', '% mix', '# of Mix nodes with k><k>', '# of mixed in top 10%(BC)', '# of mixed in bottom 10%(BC)']]


    

Data_Table_Gender.to_excel("Data table gender GoT C_IS.xlsx", index=False)

