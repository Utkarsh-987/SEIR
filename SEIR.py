''' It is an SEIR simulation based on BA Network'''

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
# Network Model i.e., BA Network

N = 50 #Population size

population_size = N

m = 5 #Number of nodes will add

initial_infectious = 1

simulation_duration = 100

# Parameters
k_E = k_I = theta_E = theta_I = 3
beta = 0.15
recovery_time = 15

# Create the network
network = nx.barabasi_albert_graph(N, m)  

# Assign initial state to nodes
states = ['S'] * population_size

# Choose random nodes initial_infectious nodes as infectious
 
for i in random.sample(range(population_size), initial_infectious):
    states[i] = 'I'

# Simulation
susceptible_count = []
exposed_count = []
infectious_count = []
recovered_count = []

for t in range(simulation_duration):

    new_states = states.copy()

    for node in network.nodes:
        if states[node] == 'S':

            for neighbor in network.neighbors(node):
                if states[neighbor] == 'I':
                    transmission_probability = np.random.exponential(beta)
                    if random.random() < transmission_probability:
                        new_states[node] = 'E'
                        break 

        elif states[node] == 'E':
            exposed_time = np.random.gamma(k_E, theta_E)
            if exposed_time >= simulation_duration:  # If exposed time exceeds simulation duration, transition to infectious
                new_states[node] = 'I'
            elif random.random() < (1 / exposed_time):
                new_states[node] = 'I'
        elif states[node] == 'I':
            if random.random() < (1 / recovery_time):
                new_states[node] = 'R'
    
    states = new_states
    
    # Count individuals in each state
    susceptible_count.append(states.count('S'))
    exposed_count.append(states.count('E'))
    infectious_count.append(states.count('I'))
    recovered_count.append(states.count('R'))

# Visualization
plt.plot(susceptible_count, label='Susceptible')
plt.plot(exposed_count, label='Exposed')
plt.plot(infectious_count, label='Infectious')
plt.plot(recovered_count, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()






