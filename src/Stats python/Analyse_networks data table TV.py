#Libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from modularity import modularity

##########
#Datasets
networks = [

    ]

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

    #G.remove_nodes_from([u for u in G if G.nodes()[u]['Sex'] == 'female']) 
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
            if u in G and len(G.nodes()[u]['gender'])==0:
                G.nodes()[u]['gender'] = gend[u]
 
    men = len([d['gender'] for (u,d) in G.nodes(True) if d['gender'] == 'male'])
    women = len([d['gender'] for (u,d) in G.nodes(True) if d['gender'] == 'female'])
    Number_Males = men
    Number_females = women
    Number_Unknown = len(G) - (women+men)
    
    Percentage_women = round((women/len(G))*100,3)
    Percentage_Unknown = round((Number_Unknown/len(G))*100,3)


    ##Male-Female Interactions

    mm = 0
    ff = 0
    mf = 0
    mu = 0
    fu = 0
    uu = 0
    for u,v in G.edges():
        if len(G.nodes()[u]) != 0 and len(G.nodes()[v]) != 0 :
            if G.nodes()[u]['gender'] == 'male' and G.nodes()[v]['gender'] == 'male':
                mm += 1
            if G.nodes()[u]['gender'] == 'female' and G.nodes()[v]['gender'] == 'female':
                ff += 1
            if G.nodes()[u]['gender'] == 'male' and G.nodes()[v]['gender'] == 'female' or G.nodes()[u]['gender'] == 'female' and G.nodes()[v]['gender'] == 'male':
                mf += 1
            if G.nodes()[u]['gender'] == 'male' and G.nodes()[v]['gender'] == '' or G.nodes()[u]['gender'] == '' and G.nodes()[v]['gender'] == 'male':
                mu += 1
            if G.nodes()[u]['gender'] == 'female' and G.nodes()[v]['gender'] == '' or G.nodes()[u]['gender'] == '' and G.nodes()[v]['gender'] == 'female':
                fu += 1
            if G.nodes()[u]['gender'] == '' and G.nodes()[v]['gender'] == '':
                uu += 1       
    FF, PercentFF = ff, round((ff/M)*100,3)
    MF, PercentMF = mf, round((mf/M)*100,3)
    FU, PercentFU = fu, round((fu/M)*100,3)
    MU, PercentMU = mu, round((mu/M)*100,3)
    MM, PercentMM = mm, round((mm/M)*100,3)
    UU, PercentUU = uu, round((uu/M)*100,3)

    female_M_No_U =  mf + ff
    female_M_G = mf+ff+fu
    male_M = mm+mu+mf
    frac_F_No_U =  round((mf + ff)/(mf+ff+mm),3)
    frac_F_G = round(((mf+ff+fu)/M)*100,3)
    frac_M_G = round(((mm+mu+mf)/M)*100,3)

    ##Important Nodes by degree

    Imp_M = 0
    Imp_F = 0
    Imp_U = 0
    for u in G.nodes:
        if G.nodes()[u]['gender'] == 'male':
            if G.degree[u] > average_degree:
                Imp_M += 1
        if G.nodes()[u]['gender'] == 'female':
            if G.degree[u] > average_degree:
                Imp_F += 1
        if G.nodes()[u]['gender'] == '':
            if G.degree[u] > average_degree:
                Imp_U += 1
    Percent_Imp_U = round((Imp_U/N)*100,3)

       
    ## Betweenness Centrality ##
    BCF = 0
    BCU = 0
    BCM = 0
    bet = nx.betweenness_centrality(G, weight='weight')
    b_sort = sorted(bet, key=bet.get,reverse=True)
    for i,u in enumerate(b_sort):
        if len(G.nodes()[u]['gender']) > 0 and G.nodes()[u]['gender'] == 'female':
            BCF += 1
        if i > (len(G))/10:
            break
    for i,u in enumerate(b_sort):
        if len(G.nodes()[u]['gender']) > 0 and G.nodes()[u]['gender'] == 'male':
            BCM += 1
        if i > (len(G))/10:
            break
    for i,u in enumerate(b_sort):
        if len(G.nodes()[u]['gender']) > 0 and G.nodes()[u]['gender'] == '':
            BCU += 1
        if i > (len(G))/10:
            break

   ### BC bottom
    UBCF = 0
    UBCU = 0
    UBCM = 0
    ub_sort = sorted(bet, key=bet.get, reverse= False)
    for i,u in enumerate(ub_sort):
        if len(G.nodes()[u]['gender']) > 0 and G.nodes()[u]['gender'] == 'female':
            UBCF += 1
        if i >= (len(G))/10:
            break
    for i,u in enumerate(ub_sort):
        if len(G.nodes()[u]['gender']) > 0 and G.nodes()[u]['gender'] == 'male':
            UBCM += 1
        if i >= (len(G))/10:
            break
    for i,u in enumerate(ub_sort):
        if len(G.nodes()[u]['gender']) > 0 and G.nodes()[u]['gender'] == '':
            UBCU += 1
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
    print(OG_Density)
    Density = round(nx.density(G),3)

    ##Average betweenness centrality of network
    bet_val = list(bet.values())
    B = np.array(bet_val)

    BC_Average = float(sum(B))/N

    ## Categorical assortativity

    Cat_S = modularity(G,'gender')

    ## assign house if unknown

    h = 0
    for n in G.nodes():
        try:
            G.nodes()[n]['house']
        except KeyError:
            h += 1
            G.nodes()[n]['house'] = f'unknown_{h}'
            
    Cat_H = modularity(G,'house')

    



    Stats_Gender = {'Narrative':Name,'N':N,'M':M,'<k>':average_degree, 'Max_degree':Max_degree, 'Average path length':path_length, 'Random path length':Random_path_length, 'Clustering':CC,
                    'Random Clustering': Random_Clustering, 'Transitivity':Transitivity, 'Assortativity':Assortativity, 'Density':Density, 'GC Size (%)':GC_size, '% Women':Percentage_women, '# Women':Number_females, 'Fraction female interaction': frac_F_No_U,
                    '% Unknown':Percentage_Unknown, '10% M&F Characters':MF_10, '# of M nodes with k><k>':Imp_M, '# of F nodes with k><k>':Imp_F,
                    '# of U nodes with k><k>':Imp_U, '% of U nodes with k><k>':Percent_Imp_U, '10% of all nodes':G_10, '# of Unknowns in top 10%(BC)':BCU, '% of Unknowns in top 10%(BC)':P_BCU, '% F-F Interactions': PercentFF,
                     '# of F-F Interactions':FF, '% F-M Interactions':PercentMF, '# of F-M Interactions':MF, '% F-Any Interactions':frac_F_G, '# of F-Any Interactions': female_M_G, '% M-Any Interactions': frac_M_G, '# of M-Any Interactions': male_M, '% M-M Interactions':PercentMM,
                    '# of M-M Interactions':MM, 'Average Betweenness':BC_Average, '# of men in top 10%(BC)':BCM, '% of men in top 10%(BC)':P_BCM, '# of men': men, '# of men in top 10%(BC)':BCM, '% of men in bottom 10%(BC)':P_UBCM,'# of men in bottom 10%(BC)':UBCM ,
                    '% of women in bottom 10%(BC)':P_UBCF,'# of women in bottom 10%(BC)':UBCF, '% of unknowns in bottom 10%(BC)':P_UBCU,'# of unknowns in bottom 10%(BC)':UBCU, '# of women in top 10%(BC)':BCF, '% of women in top 10%(BC)':P_BCF, '% of Important women as % of all women':P_BCF_of_F,'Categorical assortativity_Sex': Cat_S, 'Categorical assortativity_House': Cat_H}

    Data_Table_Gender = Data_Table_Gender.append(Stats_Gender, ignore_index=True)
    Data_Table_Gender = Data_Table_Gender[['Narrative', 'N', 'M', '<k>', 'Max_degree', 'Average path length', 'Random path length', 'Clustering', 'Random Clustering', 'Transitivity', 'Assortativity', 'Density', 'GC Size (%)', '% Women', '# Women', 'Fraction female interaction',
                    '% Unknown', '10% M&F Characters', '# of M nodes with k><k>', '# of F nodes with k><k>',
                    '# of U nodes with k><k>', '% of U nodes with k><k>', '10% of all nodes', '# of Unknowns in top 10%(BC)', '% of Unknowns in top 10%(BC)', '% F-F Interactions',
                    '# of F-F Interactions', '% F-M Interactions', '# of F-M Interactions', '% F-Any Interactions', '# of F-Any Interactions', '% M-Any Interactions', '# of M-Any Interactions', '% M-M Interactions',
                    '# of M-M Interactions', 'Average Betweenness', '# of men in top 10%(BC)', '% of men in top 10%(BC)','# of men', '% of men in bottom 10%(BC)','# of men in bottom 10%(BC)', '% of women in bottom 10%(BC)','# of women in bottom 10%(BC)', '% of unknowns in bottom 10%(BC)','# of unknowns in bottom 10%(BC)', '# of women in top 10%(BC)', '% of women in top 10%(BC)', '% of Important women as % of all women', 'Categorical assortativity_Sex','Categorical assortativity_House']]


    if i%250 == 0:
        export_excel_G = Data_Table_Gender.to_excel('Data table gender GoT TV_CE' +str(i)+'.xlsx', index=False)
        

export_excel_G = Data_Table_Gender.to_excel("Data table gender GoT TV_CE.xlsx", index=False)

