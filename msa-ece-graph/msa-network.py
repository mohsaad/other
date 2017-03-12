# Mohammad Saad
# 3/11/2017
# msa-network.py

import sys
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from random import random

# Generate a graph
G = nx.Graph()

# people dictionaries
people = {}
name_dict = {}
class_people = [[],[],[],[]]

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
        name_dict[int(data_arr[2])] = data_arr[0] + " " + data_arr[1]
        class_people[int(data_arr[3])].append(int(data_arr[2]))
        G.add_node(int(data_arr[2]))

    num_edges = int(f.readline().split("\n")[0])
    for i in range(0, num_edges):
        edge = f.readline().split("\n")[0].split(" ")
        G.add_edge(int(edge[0]), int(edge[1]))

f.close()


pos = nx.spring_layout(G)

# plot the figure
f = plt.figure()
nx.draw_networkx_nodes(G, pos = pos, nodelist = class_people[0], node_color = 'yellow', label = 'Seniors', labels = name_dict)
nx.draw_networkx_nodes(G, pos = pos, nodelist = class_people[1], node_color = 'orange', label = 'Juniors', labels = name_dict)
nx.draw_networkx_nodes(G, pos = pos, nodelist = class_people[2], node_color = 'green', label = 'Sophomores', labels = name_dict)
nx.draw_networkx_nodes(G, pos = pos, nodelist = class_people[3], node_color = 'red', label = 'Freshman', labels = name_dict)
nx.draw_networkx_edges(G, pos = pos)
nx.draw_networkx_labels(G, pos = pos, labels = name_dict, font_color = 'b')
plt.legend()
plt.title("Network of Muslim ECE Students")
plt.show()

# Some statistics
# rank the most-connected people
