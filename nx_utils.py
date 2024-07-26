import turning_function
import statistics as stat
import numpy as np

from polygon import Polygon

def network_disorder(cells, pos, n=-1):
    """
    Return the network disorder of the network.

    Parameters
    ----------
    cells : dict
        Dictionary containing cell indicies as keys and a list of cell vertices
        in counterclockwise order as values.
    pos : dict
        Dictionary containing vertex indicies as keys and an array-like object
        containing the (x,y) coordinates of each vertex as values.
    n : int, optional
        The number of sides of the regular polygon which every cell will be
        compared to. If -1 (default), every cell will be compared to a k-gon,
        where k is equal to the number of sides of each cell.

    Returns
    -------
    float
        A metric quantifying the network disorder of the network, which is the
        average turning distance of each cell with respect to the specified
        n-gon.

    """
    
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
        
        turn_dists.append(dist)

    return stat.mean(turn_dists)
        
    