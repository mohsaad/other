# Mohammad Saad
# 3/11/2017
# msa-network.py

import sys
import networkx as nx
import matplotlib.pyplot as plt

# Generate a graph
G = nx.Graph()

# people dictionaries
people = {}

# read that file
'''
Data is organized as the following:
num_people
name person_id class

We build a dictionary where
people[person_id] = (name, class)

'''
with open(sys.argv[1], 'r') as f:
    num_people = int(f.readline().split("\n")[0])
    for i in range(0, num_people):
        data_arr = f.readline().split("\n")[0].split(" ")

        people[int(data_arr[2])] = (data_arr[0] + " " + data_arr[1], int(data_arr[3]))
        G.add_node(int(data_arr[2]))

    num_edges = int(f.readline().split("\n")[0])
    for i in range(0, num_edges):
        edge = f.readline().split("\n")[0].split(" ")
        print(edge)
        G.add_edge(int(edge[0]), int(edge[1]))

f.close()

plt.figure()
nx.draw(G)
plt.show()
