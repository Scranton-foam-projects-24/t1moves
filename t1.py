import networkx as nx
# from main_file import G

#u,v are some vertices which share an edge
def t1_move(G,u,v):

    #u = left or top node
    #v = right or bottom node
    
    u_neighbors = list(nx.all_neighbors(G,u))
    v_neighbors = list(nx.all_neighbors(G,v))
    temp = 0
    #
    
    if u_neighbors[1]-u_neighbors[0] == 1: #vertical
        temp = u_neighbors[1]
        G.remove_edge(u,u_neighbors[1])
        G.add_edge(u,v_neighbors[1])
        G.remove_edge(v,v_neighbors[1])
        G.add_edge(v,temp)
    else:
        temp = u_neighbors[2]
        G.remove_edge(u,u_neighbors[2])
        G.add_edge(u,v_neighbors[0])
        G.remove_edge(v,v_neighbors[0])
        G.add_edge(v,temp)
    return G