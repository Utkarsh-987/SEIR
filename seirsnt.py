"""
SEIRS dynamics on Network models
Next-time DES method
Message Passing Theory approach
"""
import numpy as np
import pandas as pd
import networkx as nx
from ete3 import Tree
from random import sample
from funcy import lcat, get_in


def seirs_ne(g, T, params, extra=None):
    # unpack params
    con_days = params['cd']
    exp_days = params['ed']
    inf_days = params['id']
    rec_days = params['rd']
    init_inf = params['ii']
    
    # sample initially infected nodes
    init_nodes = sample(list(g.nodes), init_inf)
    
    # phylogenetic tree
    if extra=='tree':
        tree = Tree(name=init_nodes[0])
    
    # current time
    ct = 0
    
    # event list for all nodes
    events = {n:{} for n in g.nodes}
        
    # set a contact time for all edges
    for n,m in g.edges:
        g[n][m]['contime'] = int(round(np.random.exponential(con_days)))
        
    # initialize the network attributes
    for n in g.nodes:
        # for initially infected nodes
        if n in init_nodes:
            # set the status as infected
            g.nodes[n]['status'] = 'i'
            # add the infection to event list
            events[n][ct] = 'inf'
            # compute period stays in the infected state
            period = int(round(np.random.exponential(inf_days)))
            # rectime is the time the node stays in the inf state
            g.nodes[n]['rectime'] = period
            # add the recovery event to the list
            events[n][ct+period] = 'rec'
            # whether infectable now
            g.nodes[n]['flag'] = False
        else:
            # all the remaining nodes are susceptible
            g.nodes[n]['status'] = 's'
            # add being susceptible as an event
            events[n][ct] = 'sus'
            # whether infectable now
            g.nodes[n]['flag'] = True
            
            
    # run the SEIRS loop for the entire time horizon
    while ct < T:
        if extra=='gml':
            # save network for visualization
            nx.write_gml(g, f"../temp/day{ct}.gml")
        
        # get the infected nodes list
        inf_nodes = [n for n in g.nodes if g.nodes[n]['status']=='i']
        # get the list of sus neighbours of inf nodes
        sus_nodes = [g.neighbors(n) for n in inf_nodes]
        sus_nodes = [n for n in set(lcat(sus_nodes)) if g.nodes[n]['flag']]
        
        
        for n in sus_nodes:
            # infected neighbours
            inf_nbrs = [m for m in g.neighbors(n) if g.nodes[m]['status']=='i']
            # rec times of infected neighbours
            rec_times = {m:g.nodes[m]['rectime'] for m in inf_nbrs}
            # contact times sus-inf links
            con_times = {m:g[m][n]['contime'] for m in inf_nbrs}
            # check if the contact is before recovery
            inf_check = {m:con_times[m]<rec_times[m] for m in inf_nbrs}
            # if any of the time is less
            if any(inf_check.values()):
                # not infectable anymore
                g.nodes[n]['flag'] = False
                # infectionable nodes with contact times 
                causable = {m:con_times[m] for m in inf_nbrs if inf_check[m]}
                # node caused infection and contact time
                ni, et = min(causable.items(), key=lambda k: k[1])
                
                # add the exposure event to the list
                events[n][ct+et] = 'exp'
                # reset the link contact time
                g[n][ni]['contime'] = int(round(np.random.exponential(con_days))) 
                
                # add node and edge to the phylo-tree
                if extra=='tree':
                    rn = tree.search_nodes(name=ni)[-1]
                    rn.add_child(name=n)
                
        # update current time
        ct=ct+1
        # get nodes with events with current time
        nxtnodes = [n for n in g.nodes if get_in(events, [n, ct])]
        # list of events with current time
        nxtevents = [(n,get_in(events, [n, ct])) for n in nxtnodes]
        
        for n,e in nxtevents:
            if e=='exp':
                # set the status of the current node as exposed
                g.nodes[n]['status'] = 'e'
                # time stays in the exposed state
                period = int(round(np.random.exponential(exp_days)))
                # add the next event - infection - to event list
                events[n][ct+period] = 'inf'
            elif e=='inf':
                # set the status of the current node as infected
                g.nodes[n]['status'] = 'i'
                # time stays in the infected state
                period = int(round(np.random.exponential(inf_days)))
                # the rec time is the time node stays in the inf state
                g.nodes[n]['rectime'] = period
                # add the next event - recovery - to event list
                events[n][ct+period] = 'rec'
            elif e=='rec':
                # set the status of the current node as recovered
                g.nodes[n]['status'] = 'r'
                # time stays in the recovered state
                period = int(round(np.random.exponential(rec_days)))
                # add the next event - recovery - to event list
                events[n][ct+period] = 'sus'
            else:
                # set the status of the current node as susceptible
                g.nodes[n]['status'] = 's'
                # infectable now
                g.nodes[n]['flag'] = True
    
    # convert events to dataframe
    df = pd.DataFrame.from_dict(events, orient='index')
    # days less than T only
    days = set([d for d in list(df) if d<T])
    # no event days
    missd = [d for d in range(T+1) if d not in days]
    # add these cols into data
    df.loc[:,missd] = pd.NA
    # take data of the upto T days only
    df = df[list(range(T+1))].copy()
    # fill NA of every node by prev state
    df = df.ffill(axis=1)
    
    if extra=='node':
        return df
    
    if extra=='csv':
        df.to_csv(f'../temp/netd{ct}.csv', index=False)
        
    if extra=='tree':
        return tree
        #tree.write(format=1, outfile="treeson.nwk")
    
    # daily counts of states
    stats = [df[t].value_counts().to_dict() for t in range(T+1)]
    # convert into dataframe
    stats = pd.DataFrame(stats)
    # fill NAs by 0
    stats = stats.fillna(0)
    
    # change index to day column
    stats = stats.reset_index()
    stats = stats.rename(columns={'index':'day'})
    return stats
