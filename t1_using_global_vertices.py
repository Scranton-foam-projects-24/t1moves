import numpy as np
import scipy.interpolate as interpolate
import scipy as sp
import scipy.special as spec
import scipy.stats as stats
import csv
import pylab
import pyvoro

import networkx as nx

import matplotlib
import matplotlib.pyplot as plt

import random as rand

from tutte_positions import *


def voro_to_nx():
    
    # [-1,-1] is a dummy entry for ease of working with pos
    major_list = [[-1,-1]]
    local_vertices = [(-1,-1)] # this is (cell_number,local_vertex_number)
                             # index is major vertex number
    
    H = nx.Graph()
    pos = {}
    outer = []
    
    # ADDS NODES
    
    # fills major_dict
    for m in range(0,len(cells)):   # iterate over each cell
        for n in range(0,len(cells[m]['vertices'])):    # iterate over each vertex
            temp = [round(cells[m]['vertices'][n][0],7), round(cells[m]['vertices'][n][1],7)]
            # if cells[m]['vertices'][n].all != major_dict:
            if temp not in major_list:
                # major_dict.append(cells[m]['vertices'][n])
                major_list.append(temp)
                local_vertices.append((m,n))

    # print("major_list:\n"+str(major_list) + "\n")
    # print("local_vertices:\n"+str(local_vertices) + "\n")
    # print(str(len(major_list)-1) + " nodes")
    
    # fills pos for outer nodes
    for n in range(1,len(major_list)):
        H.add_node(n)
        
        if 1 in major_list[n] or 0 in major_list[n]:
            pos[n] = tuple(major_list[n])
            outer.append(n)
        
        else:
            pos[n] = (0,0)
    
    cell_major_vertices = create_cell_major_vertices(H, cells, major_list)
   
    return H, pos, major_list, outer, cell_major_vertices


def vnx_add_edges(H, major_list):
    u_pos = -1
    v_pos = -1
    
    u = -1
    v = -1
    
    for m in range(0,len(cells)):   # iterate over each cell
        for n in range(0,len(cells[m]['vertices'])):    # iterate over each vertex
            
            # get u_pos and v_pos
            # get node number from major_list
            # add edge between u and v
            if n != len(cells[m]['vertices'])-1:
                u_pos = [round(cells[m]['vertices'][n][0],7), round(cells[m]['vertices'][n][1],7)]
                v_pos = [round(cells[m]['vertices'][n+1][0],7), round(cells[m]['vertices'][n+1][1],7)]
            else:
                u_pos = [round(cells[m]['vertices'][n][0],7), round(cells[m]['vertices'][n][1],7)]
                v_pos = [round(cells[m]['vertices'][0][0],7), round(cells[m]['vertices'][0][1],7)]
            
            u = major_list.index(u_pos)
            v = major_list.index(v_pos)
            
            H.add_edge(u,v)
    
    return H


def adj_matrix_from_vnx(H):
    matrix = [[0 for n in range(0,H.number_of_nodes()+1)] for m in range(0,H.number_of_nodes()+1)]
    
    for n in range(0,H.number_of_nodes()+1):
        matrix[0][n] = n
        matrix[n][0] = n
        for m in range(1,H.number_of_nodes()+1):
            if H.has_edge(n,m):
                matrix[n][m] = 1
            else:
                matrix[n][m] = 0

    return matrix


def laplacian_from_vnx_adj(H,matrix):
    
    laplacian = matrix
    for m in range(1,len(matrix)):
        for n in range(1,len(matrix)):
            if n == m:
                laplacian[m][n] = len([x for x in H.neighbors(m)])
            elif matrix[m][n] == 1:
                laplacian[m][n] = -1

    laplacian = np.array(laplacian)
    return laplacian


def update_nx(H,adj_matrix):
    
    for n in range(1,len(adj_matrix)):
        for m in range(1,len(adj_matrix)):
            if adj_matrix[n][m] == 0 and H.has_edge(n,m):
                H.remove_edge(n,m)
            if adj_matrix[n][m] == 1 or adj_matrix[n][m] == -1:
                H.add_edge(n,m)


def update_everything(H,adj_matrix,outer,pos):
    laplacian = laplacian_from_vnx_adj(H, adj_matrix)
    io_matrix = convert_to_inner_outer(True, adj_matrix, outer)
    io_matrix = np.array(io_matrix)
    P2 = compute_positions_after_tutte(io_matrix, outer, pos)
    update_pos(P2,pos,outer)
    update_nx(H,adj_matrix)
    

def create_cell_major_vertices(H, cells, major_list):
    
    cell_major_vertices = []
    for n in range(0,len(cells)):
        cell_major_vertices.append([])
        
    for m in range(0,len(cells)): # iterate over every cell
        for n in range(0,len(cells[m]['vertices'])): # iterate over every vertex
            pos = [round(cells[m]['vertices'][n][0],7), round(cells[m]['vertices'][n][1],7)]
            if pos in major_list:
                cell_major_vertices[m].append(major_list.index(pos))
                
    return cell_major_vertices
    

def do_num_t1_moves(num):
    
    for n in range(0,num):
        
        target_1, target_2 = get_random_t1_targets(cell_major_vertices)
        do_t1_move(cell_major_vertices,major_list, target_1,target_2)
        

def get_random_t1_targets(cell_major_vertices):
    
    valid = False
    tries = 10000 # number of tries to find valid target vertices
    
    while valid == False and tries > 0:
        valid = True
        target_cell = rand.choice(cell_major_vertices)
        target_1 = rand.choice(target_cell)
        idx_1 = target_cell.index(target_1)
        idx_2 = idx_1 + 1
        if idx_2 >= len(target_cell):
            idx_2 = 0
            
        target_2 = target_cell[idx_2]
        
        # get edge cells
        it = 0
        edge_cells = []
        while it < len(cell_major_vertices):
            if (target_1 in cell_major_vertices[it]) and (target_2 in cell_major_vertices[it]):
                if len(cell_major_vertices[it]) > 3:
                    edge_cells.append(it)
            it += 1
           
        if len(edge_cells) != 2:
            valid = False
        
        vert_cells = []
        it = 0
        while it < len(cell_major_vertices):
            # target_1 xor target_2
            if (target_1 in cell_major_vertices[it]) or (target_2 in cell_major_vertices[it]):
                if not ((target_1 in cell_major_vertices[it]) and (target_2 in cell_major_vertices[it])):
                    if len(cell_major_vertices[it]) >= 3:    
                        vert_cells.append(it)
            it += 1
            
        if len(vert_cells) != 2:
            valid = False
        
    return target_1, target_2
    

def do_t1_move(cell_major_vertices,major_list, target_1,target_2):
    
    if target_1 == -1:
        if target_2 ==5 -1:
            return False
    
    # get edge cell numbers
    it = 0
    edge_cells = []
    while len(edge_cells) < 2 and it < len(cell_major_vertices):
        if (target_1 in cell_major_vertices[it]) and (target_2 in cell_major_vertices[it]):
            edge_cells.append(it)
        it += 1
    
    # get vertex cell numbers
    it = 0
    vert_cells = []
    while len(vert_cells) < 2 and it < len(cell_major_vertices):
        # target_1 xor target_2
        if (target_1 in cell_major_vertices[it]) or (target_2 in cell_major_vertices[it]):
            if not ((target_1 in cell_major_vertices[it]) and (target_2 in cell_major_vertices[it])):
                vert_cells.append(it)
        it += 1
    
    edge_neighbors(cell_major_vertices, target_1, target_2, edge_cells)
    vertex_neighbors(cell_major_vertices, target_1, target_2, vert_cells)
    
    return True
    

def edge_neighbors(cell_major_vertices, target_1, target_2, edge_cells):
    
    for n in edge_cells:
        idx_1 = cell_major_vertices[n].index(target_1)
        idx_2 = cell_major_vertices[n].index(target_2)
        
        # if the edge is normal
        if abs(idx_1 - idx_2) == 1:
            greater_idx = max(idx_1,idx_2)
            lesser_idx = min(idx_1,idx_2)
            
        # if the edge wraps around the list
        else:
            greater_idx = min(idx_1,idx_2)
            lesser_idx = max(idx_1,idx_2)
            
        greater_target = cell_major_vertices[n][greater_idx]
        lesser_target = cell_major_vertices[n][lesser_idx]
        
        # adjust adj matrix
        post_idx = greater_idx + 1
        if post_idx >= len(cell_major_vertices[n]):
            post_idx = 0
        
        post = cell_major_vertices[n][post_idx]
        adj_matrix[post][greater_target] = 0
        adj_matrix[greater_target][post] = 0
        
        # edge neighbors lose their CCW "greater" vertex
        del cell_major_vertices[n][greater_idx]
        
        
        

def vertex_neighbors(cell_major_vertices, target_1, target_2, vert_cells):
    
    # vertex neighbors discover the other target in one index greater
    #   CW from the target already known
    
    # no wraparound case necessary
    for n in vert_cells:
        if target_1 in cell_major_vertices[n]:
            idx = cell_major_vertices[n].index(target_1)
            cell_major_vertices[n].insert(idx,target_2)
            vertex_added = target_2
        else: # target_2
            idx = cell_major_vertices[n].index(target_2)
            cell_major_vertices[n].insert(idx,target_1)
            vertex_added = target_1
    
    
        # update adj matrix
        prior_idx = idx - 1
        if prior_idx < 0:
            prior_idx = len(cell_major_vertices[n])-1
            
        prior = cell_major_vertices[n][prior_idx]
        adj_matrix[prior][vertex_added] = 1
        adj_matrix[vertex_added][prior] = 1



np.random.seed([1938430])
 
# generates 10 "random" lists with 2 elements, over [0,1)
# also picks colors

dots_num = 10

colors = np.random.rand(dots_num, 3) 
points = np.random.rand(dots_num, 2)
 

# make color map (unnecessary for just random colorization)
color_map = {tuple(coords):color for coords, color in zip(points, colors)}
 
cells = pyvoro.compute_2d_voronoi(
points, # point positions, 2D vectors this time.
[[0.0, 1.0], [0.0, 1.0]], # box size
2.0, # block size
periodic = [False,False]
)

# colorize
for i, cell in enumerate(cells):    
    polygon = cell['vertices']
    plt.fill(*zip(*polygon),  color = 'black', alpha=0.1)

plt.plot(points[:,0], points[:,1], 'ko')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)

plt.show()



H, pos, major_list, outer, cell_major_vertices = voro_to_nx()
H = vnx_add_edges(H, major_list)
adj_matrix = adj_matrix_from_vnx(H)

# print(cell_major_vertices)
# for n in adj_matrix: print(n)
# print("\n")

do_num_t1_moves(100)
 
# to manually select targets for t1 moves, use the following:
# do_t1_move(cell_major_vertices,major_list)

# print(cell_major_vertices)
# for n in adj_matrix: print(n)

update_everything(H,adj_matrix,outer,pos)

# Draw graph with vertex labels
# nx.draw_networkx(H, pos, with_labels=True, node_size = 15)

# Draw graph without vertex labels
nx.draw_networkx(H, pos, with_labels=False, node_size = 0)
