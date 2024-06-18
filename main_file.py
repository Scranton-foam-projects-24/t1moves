import matplotlib.pyplot as plt
import networkx as nx
import math
import numpy as np
import copy
import random
from adj_matrix_generator import generate
from t1 import t1_move
from tutte_positions import update_neighbors,update_matrix,laplacian_from_adj,convert_to_inner_outer,compute_positions_after_tutte
from pick_random_valid_edge import pick_random_valid_edge


G = nx.Graph()

adj_matrix = generate(3,4)

# adj_matrix = [
    
#     # vertical + border
# [0, 1, 2, 3, 4, 5, 6],
# [1, 0, 1, 1, 0, 1, 0],
# [2, 1, 0, 1, 0, 0, 1],
# [3, 1, 1, 0, 1, 0, 0],
# [4, 0, 0, 1, 0, 1, 1],
# [5, 1, 0, 0, 1, 0, 1],
# [6, 0, 1, 0, 1, 1, 0],    
       
    # horizontal + border
# [0, 1, 2, 3, 4, 5, 6],
# [1, 0, 1, 1, 0, 1, 0],
# [2, 1, 0, 0, 1, 0, 1],
# [3, 1, 0, 0, 1, 1, 0],
# [4, 0, 1, 1, 0, 0, 1],
# [5, 1, 0, 1, 0, 0, 1],
# [6, 0, 1, 0, 1, 1, 0],   

# ]


adj_matrix = np.array(adj_matrix)

# Create nodes
for i in range(1,len(adj_matrix)):
    G.add_node(i)

# Add edges between nodes using adjacency matrix
n = 1
m = 1
size = len(adj_matrix)
while m < size:
    while n < size:
        if adj_matrix[n][m] == 1:
            G.add_edge(n, m)
        n += 1
    n = 1
    m += 1

# Set the outer nodes
outer = [1,2,5,6]

# Set position of outer nodes
pos = {}
outer_node = 1
for n in range(1,len(adj_matrix)):
    if n in outer:
        if outer_node == 1:
            pos[n] = (-1,1)
        elif outer_node == 2:
            pos[n] = (1,1)
        elif outer_node == 3:
            pos[n] = (-1,-1)
        else:
            pos[n] = (1,-1)
        outer_node += 1
    else:
        pos[n] = (0,0)


# example pos = {1:(-1,1), 2:(1,1), 3:(0,0), 4:(0,0), 5:(-1,-1), 6:(1,-1)}

P2 = compute_positions_after_tutte(adj_matrix, outer, pos)

plt.subplot(121)
nx.draw_networkx(G, pos, with_labels=True)
print("\nhere is the original matrix:")
print(adj_matrix)
print("")

print("here is the io matrix:")
test = convert_to_inner_outer(True, adj_matrix, outer)
print(test)

# print("\nhere is P2:")
random_edge = pick_random_valid_edge(G)
G = t1_move(G,random_edge[0],random_edge[1])
print("edge " + str(random_edge) + " was t1'd")
adj_matrix = update_matrix(G,adj_matrix)
compute_positions_after_tutte(adj_matrix,outer,pos)

plt.subplot(122)
nx.draw_networkx(G, pos, with_labels=True)

