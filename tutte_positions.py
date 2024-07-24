import networkx as nx
import numpy as np
import copy

def update_neighbors(G,n):
    n_neighbors = list(nx.neighbors(G,n))
    n_neighbors.sort()
    return n_neighbors


def update_matrix(G,adj_matrix):
    
    for n in range(1,len(adj_matrix)):
        for m in range(1,len(adj_matrix)):
            if G.has_edge(n,m):
                adj_matrix[n][m] = 1
            else:
                adj_matrix[n][m] = 0
    return adj_matrix

def laplacian_from_adj(G,adj_matrix):
    
    laplacian = copy.copy(adj_matrix)
    for n in range(1,len(adj_matrix)):
        for m in range(1,len(adj_matrix)):
            if n == m:
                laplacian[n,m] = 3     #This will probably cause a problem later, but it works for now
            elif adj_matrix[n,m] == 1:
                laplacian[n][m] = -1
            else:
                laplacian[n][m] = 0
    return laplacian


def convert_to_inner_outer(adj_to_io = bool, adj_matrix = list, outer = list): #outer is the list of outer nodes in the pre-formatted graph. For a 3x4 graph, it is [1,2,,5,6[.]]
    
    io_matrix = np.array(adj_matrix)
    TEMP = list
    TEMP = np.array(adj_matrix)
    
    #populate inner[]
    inner = []
    for n in range(1,len(adj_matrix)):
        if n not in outer:
            inner.append(n)
            
    if adj_to_io == True:
        for n in range(1,len(outer)+1):
            io_matrix[n,:] = TEMP[outer[n-1],:]
        m = 0
        for n in range(len(outer)+1,len(adj_matrix)):
            io_matrix[n,:] = TEMP[inner[m],:]
            m += 1
            
        TEMP = copy.copy(io_matrix)
        
        for n in range(1,len(outer)+1):
            io_matrix[:,n] = TEMP[:,outer[n-1]]
        
        m = 0
        for n in range(len(outer)+1,len(adj_matrix)):
            io_matrix[:,n] = TEMP[:,inner[m]]
            m += 1
            
    return io_matrix

def compute_positions_after_tutte(matrix, outer, pos):

    matrix = np.array(matrix)
    
    L2 = matrix[len(outer)+1:len(matrix),len(outer)+1:len(matrix)]
    L2_inv = np.linalg.inv(L2)
    B = np.array(matrix[len(outer)+1:len(matrix),1:len(outer)+1])
    P1 = np.array([pos[n] for n in outer])
    
    P2 = -L2_inv @ B @ P1
    
    # print("\nL2_inv:")
    # print(L2)
    # print("\nB:")
    # print(B)
    # print("\nP1:")
    # print(P1)
    # print("\nP2:")
    # print(P2)
    # print("\n")
            
    return P2

def update_pos(P2,pos,outer):
    
    p2_it = 1
    outer_it = 0
    
    while(p2_it <= len(P2)):
        if outer_it != len(outer) and p2_it + outer_it == outer[outer_it]:
            outer_it += 1
        else:
            pos[p2_it + outer_it] = P2[p2_it - 1]
            p2_it += 1
        
    # return pos


# def update_matrix_and_pos(G,matrix,P2,pos,outer):
   
#     matrix = update_matrix(G,matrix)
#     P2 = compute_positions_after_tutte(matrix, outer, pos)
#     pos = update_pos(P2,pos,outer)
    
#     return matrix, pos


def adj_to_io(adj_matrix,outer):
    
    adj_matrix = np.array(adj_matrix)
    result = laplacian_from_adj(adj_matrix)
    result = convert_to_inner_outer(True, adj_matrix, outer)
    
    return result




