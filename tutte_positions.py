import networkx as nx
import numpy as np
import copy
# from main_file import G, pos

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

def laplacian_from_adj(adj_matrix):
    
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
        
    # else:   #io to adj     not needed, but could maybe be useful
    #     for n in range(1,len(adj_matrix)):
    #         io_matrix[n] = adj_matrix[outer[n-1]]
    #     for n in range(len(outer),len(adj_matrix)):
    #         io_matrix[n] = adj_matrix[inner[n-1]]
    
    return io_matrix

def compute_positions_after_tutte(adj_matrix, outer, pos):
    matrix = laplacian_from_adj(convert_to_inner_outer(True, adj_matrix, outer))
    matrix = np.array(matrix)
    
    # print("\n")
    # print(matrix)
    
    L2 = matrix[len(outer)+1:len(matrix),len(outer)+1:len(matrix)]
    L2_inv = np.linalg.inv(L2)
    B = np.array(matrix[len(outer)+1:len(matrix),1:len(outer)+1])
    P1 = np.array([pos[n] for n in outer])
    
    P2 = -L2_inv @ B @ P1
    
    print("\nL2_inv:")
    print(L2)
    print("\nB:")
    print(B)
    print("\nP1:")
    print(P1)
    print("\nP2:")
    print(P2)
    # print("\n")
    
    # updates pos
    m = 0
    for n in range(1,len(pos)):
        if n not in outer:
            pos[n] = (P2[m][0],P2[m][1])
            m += 1
            
    return P2