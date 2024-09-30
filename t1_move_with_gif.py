import numpy as np
import pyvoro
import networkx as nx
import matplotlib.pyplot as plt
import random as rand
from tutte_positions import *

import shutil
from PIL import Image, ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES=True

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
            temp = [round(cells[m]['vertices'][n][0],10), round(cells[m]['vertices'][n][1],10)]
            if temp not in major_list:
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
    
    cell_major_vertices, vertices_in_cell = create_cell_major_vertices(H, cells, major_list)

    return H, pos, major_list, outer, cell_major_vertices, vertices_in_cell


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
                u_pos = [round(cells[m]['vertices'][n][0],10), round(cells[m]['vertices'][n][1],10)]
                v_pos = [round(cells[m]['vertices'][n+1][0],10), round(cells[m]['vertices'][n+1][1],10)]

            else:
                u_pos = [round(cells[m]['vertices'][n][0],10), round(cells[m]['vertices'][n][1],10)]
                v_pos = [round(cells[m]['vertices'][0][0],10), round(cells[m]['vertices'][0][1],10)]
            
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

    print("Done with adj_matrix_from_vnx!")
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


def update_P2_and_pos(H,io_matrix,outer,pos):

    P2 = compute_positions_after_tutte(io_matrix, outer, pos)
    pos = update_pos(P2,pos,outer)
    

def create_cell_major_vertices(H, cells, major_list):
    
    cell_major_vertices = {}
    vertices_in_cell = {}
        
    for m in range(0,len(cells)): # iterate over every cell
        for n in range(0,len(cells[m]['vertices'])): # iterate over every vertex
            pos = [round(cells[m]['vertices'][n][0],10), round(cells[m]['vertices'][n][1],10)]
            
            if pos in major_list:
                # Add vertex to cmv
                if m not in cell_major_vertices:
                    cell_major_vertices[m] = [major_list.index(pos)]
                else:
                    cell_major_vertices[m].append(major_list.index(pos))
                
                # Add vertex to vic
                if major_list.index(pos) not in vertices_in_cell:
                    vertices_in_cell[major_list.index(pos)] = [m]
                else:
                    vertices_in_cell[major_list.index(pos)].append(m)
                
    return cell_major_vertices, vertices_in_cell
    

def do_num_t1_moves(num):
    
    for n in range(0,num):
        target_1, target_2, edge_cells, vertex_cells = get_random_t1_targets(cell_major_vertices,vertices_in_cell)
        do_t1_move(cell_major_vertices,major_list, target_1,target_2, edge_cells,vertex_cells)
            
    update_P2_and_pos(H, io_matrix, outer, pos)


def get_random_t1_targets(cell_major_vertices,vertices_in_cell):
    
    valid = False
    tries = 10000 # number of tries to find valid target vertices
    
    while valid == False and tries > 0:
        valid = True
        target_cell = rand.choice(cell_major_vertices)
        if len(target_cell) < 3:
            valid = False
            continue
        target_1 = rand.choice(target_cell)
        idx_1 = target_cell.index(target_1)
        idx_2 = idx_1 + 1
        if idx_2 >= len(target_cell):
            idx_2 = 0
            
        target_2 = target_cell[idx_2]
        
        if len(vertices_in_cell[target_1]) < 2:
            valid = False
            continue
        if len(vertices_in_cell[target_2]) < 2:
            valid = False
            continue
        
        # get cells
        edge_cells = []
        vertex_cells = []
        
        for n in vertices_in_cell[target_1]:
            if n in vertices_in_cell[target_2]:
                edge_cells.append(n)
            else:
                vertex_cells.append(n)
        for n in vertices_in_cell[target_2]:
            if n not in vertices_in_cell[target_1]:
                vertex_cells.append(n)
                
        if len(edge_cells) != 2:
            valid = False
            continue
        if len(vertex_cells) != 2:
            valid = False
            continue
            
        if len(cell_major_vertices[edge_cells[0]]) <= 3:
            valid = False
            continue
        if len(cell_major_vertices[edge_cells[1]]) <= 3:
            valid = False
            continue
        
    return target_1, target_2, edge_cells, vertex_cells
    

def do_t1_move(cell_major_vertices,major_list, target_1,target_2, edge_cells,vertex_cells):
    
    edge_neighbors(cell_major_vertices, target_1, target_2, edge_cells)
    vertex_neighbors(cell_major_vertices, target_1, target_2, vertex_cells)
    
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
        
        io_post = -1
        while io_post == -1:
            io_post = np.where(io_matrix[0] == post)[0][0]
        io_gt = -1
        while io_gt == -1:
            io_gt = np.where(io_matrix[0] == greater_target)[0][0]
        io_lt = -1
        while io_lt == -1:
            io_lt = np.where(io_matrix[0] == lesser_target)[0][0]
            
        io_matrix[io_post][io_gt] = 0
        io_matrix[io_gt][io_post] = 0
        
        io_matrix[io_post][io_lt] = -1
        io_matrix[io_lt][io_post] = -1
        
        H.add_edge(post,lesser_target)
        
        # edge neighbors lose their CCW "greater" vertex
        del cell_major_vertices[n][greater_idx]
        del vertices_in_cell[greater_target][vertices_in_cell[greater_target].index(n)]
        

def vertex_neighbors(cell_major_vertices, target_1, target_2, vert_cells):
    
    # vertex neighbors discover the other target in one index greater
    #   CW from the target already known
    
    # no wraparound case necessary
    for n in vert_cells:
        if target_1 in cell_major_vertices[n]:
            vertex_present = target_1
            vertex_added = target_2
        else: # target_2
            vertex_present = target_2
            vertex_added = target_1
            
        idx = cell_major_vertices[n].index(vertex_present)
        cell_major_vertices[n].insert(idx,vertex_added)
        vertices_in_cell[vertex_added].append(n)
    
        # update adj matrix
        prior_idx = idx - 1
        if prior_idx < 0:
            prior_idx = len(cell_major_vertices[n])-1
            
        prior = cell_major_vertices[n][prior_idx]
        
        io_prior = -1
        while io_prior == -1:
            io_prior = np.where(io_matrix[0] == prior)[0][0]
        io_pres = -1
        while io_pres == -1:
            io_pres = np.where(io_matrix[0] == vertex_present)[0][0]
        io_add = -1
        while io_add == -1:
            io_add = np.where(io_matrix[0] == vertex_added)[0][0]
            
        io_matrix[io_prior][io_pres] = 0
        io_matrix[io_pres][io_prior] = 0
        
        io_matrix[io_prior][io_add] = -1
        io_matrix[io_add][io_prior] = -1
        
        H.remove_edge(vertex_present,prior)
        

def generate_gif(num_per_shot,num_t1,duration):
    
    # WARNING: WILL DELETE THE TARGET DESTINATION BEFORE EACH RUN
    
    diag_dest = "./diag/"
    area_dest = "./area/"
    log_area_dest = "./log_area/"
    edge_dest = "./edge/"
    gif_dest = "./gifs/"
    dest_list = [diag_dest,area_dest,log_area_dest,edge_dest, gif_dest]
    
    for n in dest_list:
        if os.path.exists(n):
            shutil.rmtree(n)
        os.makedirs(n)
    
    
    
    snap_num = 0
    num_t1_in_gif = 0
    diagram = []
    area = []
    log_area = []
    edges = []

    while num_t1_in_gif <= num_t1:
        
        print("Num before saving: "+str(num_t1_in_gif))
        
        snap_title = str("snap"+str(snap_num)+".png")
        
        plt.figure('nx')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        nx.draw_networkx(H, pos, with_labels=False, node_size = 0)
        plt.savefig(str(diag_dest)+str(snap_title),dpi=200)
        img = Image.open(str(diag_dest)+str(snap_title)) 
        diagram.append(img)
        plt.close()
        
        plt.figure('area')
        plt.savefig(str(area_dest)+str(snap_title),dpi=200)
        img = Image.open(str(area_dest)+str(snap_title)) 
        area.append(img)
        plt.close()
        
        plt.figure('logarea')
        plt.savefig(str(log_area_dest)+str(snap_title),dpi=200)
        img = Image.open(str(log_area_dest)+str(snap_title)) 
        log_area.append(img)
        plt.close()
        
        plt.figure('edge')
        plt.savefig(str(edge_dest)+str(snap_title),dpi=200)
        img = Image.open(str(edge_dest)+str(snap_title)) 
        edges.append(img)
        plt.close()
        
        plt.figure('nx')
        print("Doing some t1 moves...")
        do_num_t1_moves(num_per_shot)
        print("t1 moves done (for now)!")
        
        num_t1_in_gif += num_per_shot
        
        print("Doing some area...")
        areas = compute_cell_areas(pos,cell_major_vertices)
        histogram_area(areas)
        print("Done!")
        
        print("Doing some log area...")
        log_areas = compute_cell_log_areas(areas)
        histogram_log_area(log_areas)
        print("Done!")
        
        print("Doing some edges...")
        edge = compute_cell_edges(cell_major_vertices)
        histogram_edges(H,edge)
        print("Done!")
        
        snap_num += 1
        
        print("Image "+str(snap_num-1)+"/"+str(int(num_t1/num_per_shot))+" saved!")
    
    plt.figure('nx')
    diagram[0].save((str(gif_dest)+'diagram.gif'),format='GIF',append_images=diagram[0:],save_all=True,duration=duration,loop=0)
    
    plt.figure('area')
    area[0].save((str(gif_dest)+'area.gif'),format='GIF',append_images=area[0:],save_all=True,duration=125,loop=0)
    
    plt.figure('logarea')
    area[0].save((str(gif_dest)+'log_area.gif'),format='GIF',append_images=log_area[0:],save_all=True,duration=125,loop=0)
    
    plt.figure('edge')
    edges[0].save((str(gif_dest)+'edges.gif'),format='GIF',append_images=edges[0:],save_all=True,duration=125,loop=0)
    
    
def compute_cell_areas(pos,cell_major_vertices):
    
    areas_list_in_cca = []
    for m in range(0,len(cell_major_vertices)): # for each cell
        sum_of_determinants = 0
        for n in range(0,len(cell_major_vertices[m])): # for each vertex
            if n == len(cell_major_vertices[m]) - 1:
                ad = pos[cell_major_vertices[m][n]][0] * pos[cell_major_vertices[m][0]][1]
                bc = pos[cell_major_vertices[m][n]][1] * pos[cell_major_vertices[m][0]][0]
            else:
                ad = pos[cell_major_vertices[m][n]][0] * pos[cell_major_vertices[m][n+1]][1]
                bc = pos[cell_major_vertices[m][n]][1] * pos[cell_major_vertices[m][n+1]][0]
            det = ad-bc
            sum_of_determinants += det
            
        area_cp = .5 * sum_of_determinants

        areas_list_in_cca.append(area_cp)
                
    return areas_list_in_cca


def compute_cell_edges(cell_major_vertices):
    
    edge = []
    
    for n in cell_major_vertices:
        num_edges = len(cell_major_vertices[n])
        edge.append(num_edges)
        
    return edge


def compute_cell_log_areas(input_areas_list):
    
    log_areas_list = []
    
    smallest = 1
    for n in input_areas_list:
        if n < smallest:
            if n > 0:
                smallest = n
                
    for n in input_areas_list:
        if n < smallest:
            n = smallest
        log_of_area = np.log10(n)

        log_areas_list.append(log_of_area)

    return log_areas_list


def histogram_area(area_list_here):
    
    plt.figure('area')
    plt.clf()
    
    n,x,_ = plt.hist(area_list_here, bins=np.arange(-.00025,.02025,.00025), density=True, edgecolor='black', linewidth=1)
    plt.hist(area_list_here, bins=100,density=True,edgecolor='black', linewidth=1)

    plt.title('Area of Cells')
    plt.xlabel('Area')
    plt.ylabel('Density')
    
    
def histogram_log_area(data_log_areas):
    
    plt.figure('logarea')
    plt.clf()
    
    n,x,_ = plt.hist(data_log_areas, bins=np.arange(-6,-1.5,.25), density=True, edgecolor='black', linewidth=1)

    plt.title('Log of area of Cells')
    plt.xlabel('Log of area')
    plt.ylim([0,1.25])
    

def histogram_edges(H,data):
    
    plt.figure('edge')
    plt.clf()
    
    n,x,_ = plt.hist(data, bins=np.arange(3,22,1), density=True, edgecolor='black', linewidth=1)
    
    plt.title('Number of Edges of Cells')
    plt.xlabel('Edges')
    plt.xticks(np.arange(3, 21, step=2))
    plt.ylim([0,0.5])


    
if __name__ == "__main__":
        
    print("Generating Voronoi diagram...")
    plt.figure('voro')
    np.random.seed([1938430])
     
    # generates 10 "random" lists with 2 elements, over [0,1)
    # also picks colors
    
    dots_num = 2499
    
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
    
    print("Voronoi diagram created!")
    
    print("Generating nx...")
    plt.figure('nx')
    H, pos, major_list, outer, cell_major_vertices, vertices_in_cell = voro_to_nx()
    H = vnx_add_edges(H, major_list)
    print("Nx created!")
    print("Populating adjacency matrix...")
    adj_matrix = adj_matrix_from_vnx(H)
    print("Adjacency matrix created!")
    print("Populating laplacian...")
    laplacian = laplacian_from_adj(H,adj_matrix)
    print("Laplacian created!")
    print("Populating io_matrix...")
    io_matrix = convert_to_inner_outer(True, laplacian, outer)
    print("io_matrix populated!")
    
     
    
    # To manually select targets for t1 moves, use the following:
    # do_t1_move(cell_major_vertices,major_list, target_1,target_2, edge_cells,vertex_cells)
    
    plt.figure('nx')
    # print("Drawing nx...")
    # nx.draw_networkx(H, pos, with_labels=True, node_size = 15)
    # Draw graph with vertex labels
    # nx.draw_networkx(H, pos, with_labels=True, node_size = 15)
    
    print("Updating...")
    update_P2_and_pos(H,io_matrix,outer,pos)
    print("Update complete!")
    
    # Draw graph without vertex labels
    print("Setting ax...")
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    print("Ax done!")
    print("Computing areas...")
    areas_init = compute_cell_areas(pos,cell_major_vertices)
    plt.figure('area')
    histogram_area(areas_init)
    print("Done computing areas!")
    
    plt.figure('logarea')
    log_areas = compute_cell_log_areas(areas_init)
    histogram_log_area(log_areas)
    print("Done!")
    
    print("Doing some edges...")
    plt.figure('edge')
    
    edge = compute_cell_edges(cell_major_vertices)
    histogram_edges(H,edge)
    print("Done doing edges!")
    
    
    
    num_t1 = 30000
    
    # To manually select targets for t1 moves, use the following:
    # do_t1_move(cell_major_vertices,major_list, target_1,target_2, edge_cells,vertex_cells)
    
    # If you do not want a gif, uncomment this line and comment out 'generate_gif(...)'
    # do_num_t1_moves(num_t1)
    
    # How many T1 moves happen between snapshots in the GIF
    snapshot_interval = 1000
    
    print("Beginning GIF generation...")
    generate_gif(snapshot_interval,num_t1,duration=.1)
    print("GIF generation done!")
    
    plt.figure('nx')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    nx.draw_networkx(H, pos, with_labels=False, node_size = 0)
    plt.savefig('./final_fig.png',dpi=200)
    
    
