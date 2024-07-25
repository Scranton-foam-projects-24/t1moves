import turning_function
from polygon import Polygon

def nx_to_pyvoro(cells, pos, areas, n=-1):
    
    poly = Polygon()
    comp_poly = poly.regpoly(n) if n != -1 else None
    
    vertices = []
    turn_dists = []
    
    for cell in cells:
        polygon = cells[cell]
        # print("cell:", polygon)
        for i in range(len(polygon)):
            vertices.append(pos[polygon[i]])
            
        if n == -1:
            comp_poly = poly.regpoly(len(polygon))
        
        dist, _, _, _ = turning_function.distance(
            vertices, 
            comp_poly, 
            brute_force_updates=False
        )
        print(dist)
        turn_dists.append(dist)
    
    return turn_dists
        
    