import networkx as nx
import numpy as np
import random
import community
import networkx.algorithms.community as nx_comm

def modularity(G: nx.Graph, attribute: str, normalise = True):
    """
    Takes a graph and an attribute (cat variable). 

    Returns the modularity of the graph based on that variable
    """
    classifier = set(nx.get_node_attributes(G, name = attribute).values())
    Q = 0
    a_rsq = 0
    m = len(G.edges())
    for i in classifier:
        e_rr = 0
        a_r = 0
        for edge in G.edges():
            if G.nodes()[edge[0]][attribute] == i and G.nodes()[edge[1]][attribute] == i: 
                e_rr += 1
        for node in G.nodes():
            if G.nodes()[node][attribute] == i:
                a_r += G.degree(node)
        e_rr /= m
        a_r /= 2 * m
    Q += (e_rr - a_r**2)
    a_rsq += a_r**2
    if normalise == True:
        rho = Q / (1 - a_rsq)
        if Q >= 0:
            return rho
        if Q < 0:
            rho_min = ((-1 * a_rsq)/(1 - a_rsq))
            return -1*rho/rho_min
    else:
        return Q 

if __name__ == '__main__':
#        
