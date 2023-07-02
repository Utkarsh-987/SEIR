"""
small world network models
pref attachment network models
stoch block network models
"""
import numpy as np
import networkx as nx
from itertools import combinations
from random import choice, choices, random

def rewire(G, p):
    for u,v in G.edges:
        if random() < p:
            choosable = [n for n in G.nodes if n not in G[u]]
            choosable.remove(u)
            new_neib = choice(choosable)
            G.add_edge(u, new_neib)
            G.remove_edge(u,v)
            
    return G


def trian2d(popn):
    n = int(round(np.sqrt(popn)))
    m = 2*n-1
    k = int(popn/m)
    
    edges = []
    
    for i in range(k+1):
        for j in range(i*m,(i+1)*m):
            if j==i*m:
                edges.append((j,j+n-1))
            elif j==(i*m + n - 1):
                edges+=[(j,j+n-1),(j,j+n)]
            elif j==(i*m + n):
                edges+=[(j,j+1),(j,j+n)]
            else:
                edges+=[(j,j+1),(j,j+n-1),(j,j+n)]
                
                
    edges = [e for e in edges if max(e)<=popn and min(e)>0]
    # create a graph object        
    G = nx.Graph()
    # add edges
    G.add_edges_from(edges)
    
    return G

def swtl(n, p):
    '''
    Small-world network using triangular lattice

    Parameters
    ----------
    popn : int
        num of nodes.
    p : float
        rewire probability.

    Returns
    -------
    G : networkx graph object
        a watts-stogatz network.

    '''
    G = trian2d(n)
    G = rewire(G, p)
    return G

def set_age(G):
    for n in G.nodes():
        G.nodes[n]['age']= G.nodes[n]['age']+1
    return G

def comp_probs(G,m):
    nodes = list(G.nodes())
    degrees = [G.degree[n] for n in nodes]
    ages = [G.nodes[n]['age'] for n in nodes]
    deg_ages = [degrees[n]/ages[n] for n in range(len(nodes))]
    probs = [da/sum(deg_ages) for da in deg_ages]
    return probs
        
    
def panwat(n, m):
    # pref attachment model with ageing and triads
    
    # start with m nodes
    G = nx.empty_graph(m)
    
    # set age
    for v in G.nodes():
        G.nodes[v]['age']=0
    
    # list of nodes
    nodes = list(G.nodes())
    # since nobody has links, assign equal weightage
    probs = [1/len(nodes) for k in nodes]
    
    for i in range(m,n):
        # update age
        G = set_age(G)
        # select m num of nodes
        select = choices(nodes, weights=probs, k=m)
        # create edge list
        edges = [(i,s) for s in select]
        # add edges to the nodes
        G.add_edges_from(edges)
        # set age for new node
        G.nodes[i]['age']=1
        
        # triad formation
        # choose one node from the selected
        chosen = choice(select)
        # randomly select one of neighbors of the chosen
        ch_nbr = choice(list(G.neighbors(chosen)))
        
        # the add a link to the neighbor if possible
        if ch_nbr!=i and not G.has_edge(i, ch_nbr):
            G.add_edge(i, ch_nbr)
        
        # list of nodes
        nodes = list(G.nodes())
        # recompute node probabilities for next iter
        probs = comp_probs(G,m)
        
    return G

def sbm(n_comms, node_comms, inter_prob, intra_prob):
    # stochastic block model
    # number of nodes
    n_nodes = len(node_comms)
    
    # network connectivity matrix
    edges = np.zeros((n_nodes,n_nodes))
    
    # iterate over nodes and form edges
    for u,v in combinations(range(n_nodes),2):
        if node_comms[u]==node_comms[v]:
            if random() < intra_prob:
                edges[u,v] = 1
        elif random() < inter_prob:
            edges[u,v] = 1
            
    
    # convert upper triangular matrix to symmetic
    edges = edges + edges.T
    # SBM from edge matrix
    G = nx.from_numpy_array(edges)
    # set community name as block attribute
    for n in G.nodes:
        G.nodes[n]['block'] = node_comms[n]
    return G

def sfcomm(n, c, k, p):
    '''
    Parameters
    ----------
    n : int
        number of nodes.
    c : int
        number of communities.
    k : int
        average degree.
    p : float
        rewire probability for inter community links.

    Returns
    -------
    G : nx.Graph
        scale free network with communities.
    '''
    # generate a BA network
    G = nx.barabasi_albert_graph(n, k)
    # compute communities using greedy method
    comms = nx.community.greedy_modularity_communities(G, cutoff=c, best_n=c)
    
    # iterate over communities
    for i, cm in enumerate(comms):
        # loop over nodes in the community
        for n in cm:
            # add community information
            G.nodes[n]['community'] = i
            # outside community links
            inter = [m for m in G[n] if m not in cm]
            # for every outside community link
            for m in inter:
                # break the link with a probability
                if random() < p:
                    # find nodes from same community to link
                    linkable = [r for r in cm if r not in G[n]]
                    linkable.remove(n) # to avoid self-loops
                    # if any linkable node available, break old
                    if any(linkable):
                        G.remove_edge(n, m)                        
                        # add one randomly
                        G.add_edge(n, choice(linkable))
                    
    return G
    

# Code taken from https://github.com/ftudisco/scalefreechunglu
# added self loop removal
def make_cl_graph(w):
    # Outputs the networkx.Graph of the graph
    n = np.size(w)
    s = np.sum(w)
    m = ( np.dot(w,w)/s )**2 + s
    m = int(m/2)
    wsum = np.cumsum(w)
    wsum = np.insert(wsum,0,0)
    wsum = wsum / wsum[-1]
    I = np.digitize(np.random.rand(m,1),wsum)
    J = np.digitize(np.random.rand(m,1),wsum)
    G = nx.Graph()
    G.add_nodes_from(range(1,n+1))
    G.add_edges_from(tuple(zip(I.reshape(m,),J.reshape(m,))))
    
    # remove self edges and added one edge randomly
    for u, v in nx.selfloop_edges(G):
        G.remove_edge(u, v)                        
        linkable = [n for n in G.nodes if n not in G.neighbors(u)]
        if any(linkable):
            G.add_edge(u, choice(linkable))
            
    return G

def chunglusf(n, d, gamma):
    m = n ** .5 
    p = 1/(gamma-1)
    c = (1-p)*d*(n**p)
    i0 = (c/m)**(1/p) - 1
    w = [c/((i+i0)**p) for i in range(n)]    
    return make_cl_graph(w)

