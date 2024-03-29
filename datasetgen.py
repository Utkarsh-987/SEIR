#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
If libraries are missing. Run these commands on your notebook 
for google colab following install the following libraries
  !pip install torch_geometric --quiet
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
import random as rd
from torch_geometric.data import Data

from seirsnt import seirs_ne, state_ohe
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
class Epidata:
  def __init__(self,g,x,params,epi_params,net_name,combo_class):
    self.adj = nx.adjacency_matrix(g).toarray()
    self.data = x
    self.Epidemic_Parameters = {'Contact Days':params['cd'],'Exposed Days':params['ed'],'Infected Days':params['id'],
                                'Recovery Days':params['rd'],'Initial Infections':params['ii']}
    self.net_name = net_name
    self.cname = combo_class
# data generation for ER network
# 16 possible combos
combos = list(product([0, 1], repeat=4))
# iterate over combinations
# save data
cdata = []
for comb in combos:
    # print combo name
    cname = ''.join([str(d) for d in comb])
    # print(datetime.now(), cname)

    # get seirs param rages for the combo
    for n in range(12):
          idcap = rd.randint(*id_space[comb[0]])
          edcap = rd.randint(*ed_space[comb[1]])
          rdcap = rd.randint(*rd_space[comb[2]])
          ircap = rd.uniform(*ir_space[comb[3]])

          # iterate over params and do simulation
          i,j,k,l = (idcap, edcap, rdcap, ircap)
          # average contact time of an edge
          cd = round(1/l, 2)
          # all parameters
          params = {'id':i, 'ed':j, 'rd':k,'ir':l, 'cd':cd, 'ii':init_inf}
          # generate network
          g = nx.gnp_random_graph(n=nodes, p=linkp)
          # do SEIRS sim
          df = seirs_ne(g, T, params, extra='node')
          # one hot encoding
          x = state_ohe(df)
          D = Epidata(g,x,params,avg_deg,"ER Network",cname)
          cdata.append(D)
with open("ER Dataset.pickle",'ab') as file:
  my_pickle = pickle.dump(cdata,file)
        # save x, g as adj matrix, network name, params, epidemic params, combo class
# Link of google colab to run and download dataset for all six network
# https://colab.research.google.com/drive/17J7gb2Sfj3ZalLZcC1vH5sv_gl3842g_?usp=sharing
