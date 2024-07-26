import turning_function
import statistics as stat
import numpy as np

from polygon import Polygon

def nx_to_pyvoro(cells, pos, n=-1):
    
    poly = Polygon()
    comp_poly = poly.regpoly(n) if n != -1 else None
    
    turn_dists = []
    
    for cell in cells:
        polygon = cells[cell]
        vertices = []
        for i in range(len(polygon)):
            vertices.append(np.array(pos[polygon[i]]))
            
        if n == -1:
            comp_poly = poly.regpoly(len(polygon))
        
        dist, _, _, _ = turning_function.distance(
            vertices, 
            comp_poly, 
            brute_force_updates=False
        )
        
        # print(polygon, dist)
        turn_dists.append(dist)

    return stat.mean(turn_dists)
        
    