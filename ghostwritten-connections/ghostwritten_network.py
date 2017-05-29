#!/usr/bin/env python

# Mohammad Saad
# 5/27/2017
# ghostwritten_connections.py
# A visualization of the different ways the novel "Ghostwritten" has
# connections between the main stories. (Excluding the last one).

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

chapter_titles = {}

G = nx.Graph()


G.add_node(0, {"title":"Okinawa"})
G.add_node(1, {"title":"Tokyo"})
G.add_node(2, {"title":"Hong Kong"})
G.add_node(3, {"title":"Holy Mountain"})
G.add_node(4, {"title":"Mongolia"})
G.add_node(5, {"title":"Petersburg"})
G.add_node(6, {"title":"London"})
G.add_node(7, {"title": "Clear Island"})
G.add_node(8, {"title": "Night Train"})

# add to dictionary
chapter_titles[0] = "Okinawa"
chapter_titles[1] = "Tokyo"
chapter_titles[2] = "Hong Kong"
chapter_titles[3] = "Holy Mountain"
chapter_titles[4] = "Mongolia"
chapter_titles[5] = "Petersburg"
chapter_titles[6] = "London"
chapter_titles[7] = "Clear Island"
chapter_titles[8] = "Night Train"



# Connections in chapter one
G.add_edge(0, 1, description="The dog needs to be fed.")

# Connections in chapter two
G.add_edge(1, 2, description="Satoru and Tomoyo in Hong Kong")

# Connections in Chapter 3
G.add_edge(2, 3, description="The Maid")

# connections in chapter 4
G.add_edge(3, 4, description="The Three Animals")

# connections in chapter 5
G.add_edge(4, 5, description="Suhbataar")

# connections in chapter 6
G.add_edge(5, 6, description="Jerome")
G.add_edge(5, 2, description="Account #1390931")

# connections in chapter 7
G.add_edge(6, 2, description="Katy Forbes")
G.add_edge(6, 7, description="Mo Muntervary")

# connections in chapter 8
G.add_edge(7, 2, description="Huw Llewellyn")
G.add_edge(7, 8, description="The Zookeeper")

# connections in chapter 9
G.add_edge(8, 0, description="Quasar calling in")
G.add_edge(8, 4, description="Noncorpum")

#node_colors = np.linspace(0,1,1.0/len(G.nodes()))

pos = nx.circular_layout(G)
plt.figure()
nx.draw_networkx_nodes(G, pos = pos)
nx.draw_networkx_edges(G, pos = pos)
nx.draw_networkx_labels(G, pos = pos, labels=chapter_titles)
plt.show()
