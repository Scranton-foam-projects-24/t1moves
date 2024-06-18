import random

def pick_random_valid_edge(G):
    valid_edge = False
    while valid_edge == False:
        u = random.randint(1,G.number_of_nodes())
        if len(list(G.neighbors(u))) >= 3:
            v = random.choice(list(G.neighbors(u)))
            if len(list(G.neighbors(v))) >= 3:
                valid_edge = True
            else:
                valid_edge = False
    result = sorted((u,v))
    return result