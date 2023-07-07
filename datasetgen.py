#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
If libraries are missing. Run these commands on your notebook 
for google colab following install the following libraries
  !pip install torch_geometric --quiet
  !pip install dfutils --quiet
  !pip install ete3 --quiet
  !pip install funcy --quiet
"""

import warnings
import lzma, pickle
from itertools import product
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data

from seirsnt import seirs_ne
# from dfutils import state_ohe // this is missing so I have created a custom ohe
from netmodels import panwat, swtl, chunglusf, sbm, sfcomm
#------------------------------------------------------------------------------#
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# time horizon
T = 100

# num of sim runs
N = 40
batch_size = 50

# Network properties
nodes = 1000
avg_deg = 4 
rw_prob = 0.1 # rewire probability
plexp = 3 # cl power law exponent
cldeg = 10 # cl avg degree
linkp = 0.01 # link probability

communes = 10 # number of communities
intra_prob = 0.05 # SBM intra-block link prob
inter_prob = 0.01 # SBM inter-block link prob


#------------------------------------------------------------------------------#
# SEIRS params ranges
init_inf = 1

# infection period
ids_min, ids_max = 1, 7
idl_min, idl_max = 10, 30
# exposed period
eds_min, eds_max = 1,7
edl_min, edl_max = 10,21
# recovery period
rds_min, rds_max = 90, 180
rdl_min, rdl_max = 360, 720
# infection rate
irs_min, irs_max = 0.01, 0.3
irl_min, irl_max = 0.5, 0.9

# param divisions
divide = 5

# create range list for all params
id_space = [(ids_min, ids_max), (idl_min, idl_max)]
ed_space = [(eds_min, eds_min), (edl_min, edl_max)]
rd_space = [(rds_min, rds_max), (rds_min, rds_max)]
ir_space = [(irs_min, irs_max), (irl_min, irl_max)]
#------------------------------------------------------------------------------#

# six network models

# BA network
g = panwat(n=nodes, m=avg_deg)

# WS network
g = swtl(n=nodes, p=rw_prob)

# CL network
g = chunglusf(n=nodes, d=cldeg, gamma=plexp)

# ER network
g = nx.gnp_random_graph(n=nodes, p=linkp)

# SBM network
node_comms = [i % communes for i in range(nodes)]
g = sbm(n_comms=communes, node_comms=node_comms,
                inter_prob=inter_prob, intra_prob=intra_prob)

# Scale free community network
g = sfcomm(n=nodes, c=communes, k=avg_deg, p=rw_prob)
#------------------------------------------------------------------------------#

# Class to save a generated dataset as an object in pickle
class Dataset:
  def __init__(self,g,x,params,epi_params,net_name,combo_class):
    self.Network_adj_matrix = nx.adjacency_matrix(g).toarray()
    self.Population_adj_matrix = x
    # self.Network_Parameters = epi_params
    self.Epidemic_Parameters = {'Contact Days':params['cd'],'Exposed Days':params['ed'],'Infected Days':params['id'],
                                'Recovery Days':params['rd'],'Initial Infections':params['ii']}
    self.Network_name = net_name
    self.Combination_class = cname 
    
# Code for one_hot_encoding
def one_hot_encode(variable):
    encoding = {
        'sus': '000',
        'exp': '001',
        'inf': '010',
        'rec': '100'
    }

    if variable in encoding:
        return encoding[variable]
    else:
        return '0000'  # Default encoding for unknown values

# data generation for ER network

# 16 possible combos
combos = list(product([0, 1], repeat=4))

# iterate over combinations
for comb in combos:
    # print combo name
    cname = ''.join([str(d) for d in comb])
    # print(datetime.now(), cname)
    
    # save data
    # cdata = []
    
    # get seirs param ranges for the combo
    idcap = np.linspace(*id_space[comb[0]], divide)
    edcap = np.linspace(*ed_space[comb[1]], divide)
    rdcap = np.linspace(*rd_space[comb[2]], divide)
    ircap = np.linspace(*ir_space[comb[3]], divide)
    # iterate over params and do simulation
    for i,j,k,l in product(idcap, edcap, rdcap, ircap):
        # average contact time of an edge
        cd = round(1/l, 2)
        # all parameters
        params = {'id':i, 'ed':j, 'rd':k,'ir':l, 'cd':cd, 'ii':init_inf}
        
        # generate network
        g = nx.gnp_random_graph(n=nodes, p=linkp)
        # do SEIRS sim
        df = seirs_ne(g, T, params, extra='node')
        # one hot encoding 
        x = df.applymap(one_hot_encode)
        D = Dataset(g,x,params,avg_deg,'BA Network',cname)
        with open('BA Dataset.pickle','ab') as f:
          my_pickle = pickle.dump(D,f)
        # save x, g as adj matrix, network name, params, epidemic params, combo class
