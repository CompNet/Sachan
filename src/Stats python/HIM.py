import networkx as nx
import numpy as np
from scipy import integrate


def Hamming(G, H):
    """
    Takes two networkx graph objects, returns the hamming distance between them.
    """
    if len(G.nodes()) != len(G.nodes()):
        print('Hamming distance requires graphs of same size')
    
    A1 = nx.adjacency_matrix(G)
    A2 = nx.adjacency_matrix(H)
    N = len(G.nodes())
    return ((A1 - A2).sum())/(N * (N - 1))


def IM(G, H, gamma):
    """
    Calculates the Ipsen Mikhailov distance between two graphs. Graphs need not be the same size.
    
    Parameters
    ----------

    G, H: networkx graph
    
    gamma: float

    """
    L_g = nx.laplacian_spectrum(G)
    L_h = nx.laplacian_spectrum(H)
    L_g[L_g < 10**-6] = 0
    L_h[L_h < 10**-6] = 0

    omega_g = np.sqrt(L_g)
    omega_h = np.sqrt(L_h)

    Lorentz = lambda omega, w, gamma:  np.sum(gamma * (1/((omega - w)**2 + gamma**2)))

    K_g, err = (integrate.quad(Lorentz, 0, np.inf, args = (omega_g, gamma)))
    K_h, err = (integrate.quad(Lorentz, 0, np.inf, args = (omega_h, gamma)))
    K_g = 1/K_g
    K_h = 1/K_h

    difference = lambda omega: (K_g * Lorentz(omega, omega_g, gamma) - K_h * Lorentz(omega, omega_h, gamma))**2
    IM, err = np.sqrt(integrate.quad(difference, 0, np.inf))

    return np.sqrt(IM)



def gamma(G1,G2,n_decimal=2,lower=0.1,upper=0.7):
    """gamma of Lorentz distributions
    This function calculates the gamma parameter for a complete and empty network of
    the same size. Note the choice of the larger one to be empty to reduce computation time

    It takes a value from 0 to 1.
    TO DO: Put in a break so that it stops sweeping for gamma after a local extrama close to 1

    Parameters
    ----------
    G1, G2 : networkx graph object
        Networks to compare
    n_decimal : int (optional, default: 2)
        The number of decimal places we want gamma, it doesn't need to be particularly high
    lower : float (optional, default: 0.1)
        The lower bound for where we want to begin looking for gamma
    upper : float (optional, default: 0.7)
        The upper bound for where we stop looking for an optimal gamma
        
    Returns
    -------
    gam : float
    
    """
    n1, n2 = len(G1), len(G2)
    if n1 <= n2:
        F = nx.complete_graph(n1)
        E = nx.empty_graph(n2)
    else:
        F = nx.complete_graph(n2)
        E = nx.empty_graph(n1)

    s = '.'
    for i in range(n_decimal):
        s += '9'
    
    for gam in range(int(lower*(10**n_decimal)),int(upper*(10**n_decimal))):
        gam = gam/(10**n_decimal)
        d = IM(F,E,gam)
        if round(d,n_decimal) == float(s):
            break
    return gam

def gamma_s(G, H):
    """
    Gets the appropriate value for gamma.
    """    
    n1 = len(G.nodes())
    n2 = len(H.nodes())
    E = nx.empty_graph(n1)
    F = nx.complete_graph(n2)


    for scale in range(1, 70):
        gamma = scale/100
        d = IM(E, F, gamma)
#        print(np.round(d - 1, 2))
        if np.round(d - 1, 2) == 0:
            break
    
    return gamma


    #for gamma in some range:
    #calculate IM between full and empty graph above. When gamma gives approximately 1, that is the gamma value.

if __name__ == '__main__':
    G = nx.complete_graph(100)
    H = nx.empty_graph(100)
    d = IM(G, H, gamma(G,H))
    print(d)