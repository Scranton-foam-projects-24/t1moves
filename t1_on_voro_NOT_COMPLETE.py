import numpy as np
import scipy.interpolate as interpolate
import scipy as sp
import scipy.special as spec
import scipy.stats as stats
import csv
import pylab
import pyvoro

import networkx as nx
import copy
import sys

import matplotlib
import matplotlib.pyplot as plt

import random as rand

from tutte_positions import *

def voro_to_nx():
    
    # [-1,-1] is a dummy entry for ease of understanding pos
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

    # fills pos for outer nodes
    for n in range(1,len(major_list)):
        H.add_node(n)
        
        if 1 in major_list[n] or 0 in major_list[n]:
            pos[n] = tuple(major_list[n])
            outer.append(n)
        
        else:
            pos[n] = (0,0)
        
    # nx.draw_networkx(H, pos, with_labels=True)
    
    return H, pos, major_list, outer


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
                # temp = len([x for x in H.neighbors(m)])
                # laplacian[m][m] = temp
                laplacian[m][n] = len([x for x in H.neighbors(m)])
            elif matrix[m][n] == 1:
                laplacian[m][n] = -1

    laplacian = np.array(laplacian)
    return laplacian

def update_everything(H,adj_matrix,outer,pos):
    # adj_matrix = update_matrix(H,adj_matrix)
    voro_to_nx()
    adj_matrix = adj_matrix_from_vnx(H)
    laplacian = laplacian_from_vnx_adj(H, adj_matrix)
    io_matrix = convert_to_inner_outer(True, adj_matrix, outer)
    io_matrix = np.array(io_matrix)
    P2 = compute_positions_after_tutte(io_matrix, outer, pos)

    update_pos(P2,pos,outer)


def do_num_t1_moves(cells,num):
    
    for n in range(0,num):
        
        loop = True
        while loop == True:
            tgt = get_random_t1_target(cells)
            if tgt[0] == True:
                loop = False
        cell = tgt[1]
        edge = tgt[2]
        do_t1_move_on_voro(cells, cell, edge)
        n += 1
    

def get_random_t1_target(cells):
    
    result = True
    
    cell = cells.index(rand.choice(cells))
    edge = cells[cell]['faces'].index(rand.choice(cells[cell]['faces']))
    
    adj_cell = cells[cell]['faces'][edge]['adjacent_cell']
    
    # get adj_edge
    # for n in cells[adj_cell]['faces']:
    #     if n['adjacent_cell'] == cell:
    #         adj_edge = cells[adj_cell]['faces'].index(n)
    
    # list of both the end neighbors
    off_neighbors = []
    for n in cells[cell]['faces']:
        for m in cells[adj_cell]['faces']:
            if n['adjacent_cell'] == m['adjacent_cell']:
                off_neighbors.append(n['adjacent_cell'])
    
    if len(off_neighbors) < 2:
        return False, cell, edge
    
    cell_list = [cell, adj_cell, off_neighbors[0], off_neighbors[1]]
    
    for n in cell_list:
        if n < 0 or len(cells[n]['faces']) < 3:
            result = False
    
    return result, cell, edge
    
    

def do_t1_move_on_voro(cells, cell, edge):
    
    # issue with (6,3) then (5,2) on 1938430 with 10 cells
    
    # cell = 5
    # edge = 2
    # cells are 6, 5, 8, 3
    
    adj_cell = cells[cell]['faces'][edge]['adjacent_cell']
    
    # get adj_edge
    for n in cells[adj_cell]['faces']:
        if n['adjacent_cell'] == cell:
            adj_edge = cells[adj_cell]['faces'].index(n)
    
    # list of both the end neighbors
    off_neighbors = []
    for n in cells[cell]['faces']:
        for m in cells[adj_cell]['faces']:
            if n['adjacent_cell'] == m['adjacent_cell']:
                off_neighbors.append(n['adjacent_cell'])
                
    on_edge(cell,edge)
    on_edge(adj_cell,adj_edge)
    
    off_edge(off_neighbors[0],cell,adj_cell,off_neighbors)
    off_edge(off_neighbors[1],cell,adj_cell,off_neighbors)    
    

def get_target_off_edge(target_cell,cell,adj_cell,off_neighbors):
    e1 = -1 # lower edge
    e2 = -1 # upper edge
    target_vertex = -1
    for n in cells[target_cell]['faces']:
        if n['adjacent_cell'] == cell:
            e1 = cells[target_cell]['faces'].index(n)
        if n['adjacent_cell'] == adj_cell:
            e2 = cells[target_cell]['faces'].index(n)
    
    for n in cells[target_cell]['faces'][e1]['vertices']:
        if n in cells[target_cell]['faces'][e2]['vertices']:
            target_vertex = n
        
    return target_vertex

    
def off_edge(target_cell,cell,adj_cell,off_neighbors):
    
    # get target
    target_vertex = get_target_off_edge(target_cell,cell,adj_cell,off_neighbors)
    
    # add vertex
    new_vertex = target_vertex + 1
    if new_vertex > len(cells[target_cell]['vertices']):
        new_vertex = 0
    cells[target_cell]['vertices'].insert(new_vertex,np.array([0.0001,0.0001]))
    
    # increment vertices
    for n in cells[target_cell]['faces']:
        if abs(n['vertices'][0] - n['vertices'][1]) == 1:
            if max(n['vertices']) != target_vertex:
                for m in n['vertices']:
                    if m >= target_vertex:
                        n['vertices'][n['vertices'].index(m)] = m + 1
        else:
            for m in n['vertices']:
                if m > target_vertex-1:
                    n['vertices'][n['vertices'].index(m)] = m + 1
    
    # add edge
    for n in off_neighbors:
        if n != target_cell:
            new_neighbor = n 
    new_edge = {'adjacent_cell':new_neighbor, 'vertices':[target_vertex,new_vertex]}
    cells[target_cell]['faces'].append(new_edge)
    
    dest_cell = -1
    newer_vertex = new_vertex + 1
    if newer_vertex >= len(cells[target_cell]['vertices']):
        newer_vertex = 0
    for n in cells[target_cell]['faces']:
        if new_vertex in n['vertices']:
            if newer_vertex in n['vertices']:
                dest_cell = n['adjacent_cell']
    
    dest_edge = -1
    for n in cells[dest_cell]['faces']:
        if n['adjacent_cell'] == target_cell:
            dest_edge = cells[dest_cell]['faces'].index(n)
    
    dest_target = get_target(dest_cell,dest_edge)
    if dest_target > len(cells[cell]['vertices']):
        dest_target = 0
        
    cells[target_cell]['vertices'][new_vertex] = cells[dest_cell]['vertices'][dest_target]

    

    
def on_edge(cell,edge):
    
    target = get_target(cell,edge)
    
    # target = target - 1
    # if target == -1:
    #     target = len(cells[cell]['vertices'])
    
    # remove edge
    del cells[cell]['faces'][edge]

    # adjust vertex positions
    adjust = target - 1
    if adjust < 0:
        adjust = len(cells[cell]['vertices'])-1
    cells[cell]['vertices'][adjust] = cells[cell]['vertices'][target]
    
    # remove target
    del cells[cell]['vertices'][target]
    
    # adjust edges
    for n in cells[cell]['faces']:
        for m in n['vertices']:
            if m >= target:
                n['vertices'][n['vertices'].index(m)] = m - 1
    
    
def get_target(cell,edge):
    
    vertices = cells[cell]['faces'][edge]['vertices']
    if abs(vertices[0] - vertices[1]) == 1:
        edge_vertex = max(vertices)
    else:
        edge_vertex = 0
    target = edge_vertex

    return target

    
def pick_t1_edge():
    
    cell = cells.index(rand.choice(cells))
    edge = -1
    if len(cells[cell]['faces']) < 3:
        cell = -1
    else:
        edge = rand.choice([x for x in range(0,len(cells[cell]['faces']))])
    
    return cell, edge
    



    

np.random.seed([1938431]) # was 1938430, with dots_num = 10
 
# generates 10 "random" lists with 2 elements, over [0,1)
# also picks colors

dots_num = 4
# dots_num = 100

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




# do_num_t1_moves(cells,1)
# do_num_t1_moves(cells,1)

"""
first do (2,4)
cell 1 is messed up
    vertex 0 is bottom right corner

edge 5 has wrong adjacent cell!
    edge 5 is between vertices 2,3

adj_cell of (1,5) == -4 (should be 0)

"""

do_t1_move_on_voro(cells, 2, 4)
# do_t1_move_on_voro(cells, 8, 0)





# colorize
for i, cell in enumerate(cells):    
    polygon = cell['vertices']
    plt.fill(*zip(*polygon),  color = 'black', alpha=0.1)


plt.plot(points[:,0], points[:,1], 'ko')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)

plt.show()

H, pos, major_list, outer = voro_to_nx()
H = vnx_add_edges(H, major_list)

adj_matrix = adj_matrix_from_vnx(H)
update_everything(H,adj_matrix,outer,pos)


nx.draw_networkx(H, pos, with_labels=True, node_size = 15)

