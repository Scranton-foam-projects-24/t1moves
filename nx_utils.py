import turning_function
import numpy as np
import math

from polygon import Polygon

def has_overlapping_vertices(vertices):
    """
    Returns whether the provided list of vertices contains two vertices located
    at the same position.

    Parameters
    ----------
    vertices : list
        List of array-like objects containing the (x,y) coordinates of each
        vertex as values.

    Returns
    -------
    bool
        The result of comparing the number of unique x-values with the total
        number of x-values and the number of unique y-values with the total
        number of y-values. 

    """
    for i, vertex1 in enumerate(vertices):
        for j, vertex2 in enumerate(vertices):
            if (round(vertex1[0],10) == round(vertex2[0],10) and
                round(vertex1[1],10) == round(vertex2[1],10) and
                i != j):
                return True
    return False



#given cumulative perimeters and angles,  here's the formula for circle distance

def circ_dist( p, theta):
    
    n = len(p)
    s1 = [  (2*p[i]- theta[i-1]/np.pi)**3 - (2*p[i-1]- theta[i-1]/np.pi)**3 for i in range(1,n)  ]
    s2 = [  theta[i-1]*(p[i]- p[i-1])/np.pi for i in range(1,n)]
    return( np.pi*np.sqrt( 1/6*sum(s1)- (1-sum(s2))**2  )        )


def cca(a,b,c):
    """Calculates the counterclockwise angle between three points."""

    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return(np.pi - angle)












def network_disorder(cells, pos, n=-1, areas=None):
    """
    Return the network disorder of the network.

    Parameters
    ----------
    cells : dict
        Dictionary containing cell indices as keys and a list of cell vertices
        in counterclockwise order as values.
    pos : dict
        Dictionary containing vertex indices as keys and an array-like object
        containing the (x,y) coordinates of each vertex as values.
    n : int, optional
        The number of sides of the regular polygon which every cell will be
        compared to. If -1 (default), every cell will be compared to a k-gon,
        where k is equal to the number of sides of each cell.
    areas : list, optional
        Optional argument containing the list of cell areas to be used when
        weighting computed turning distances. If None (default), the turning
        distance of each cell will not be multiplied by the cell area.

    Returns
    -------
    list
        A list containing the turning distance of each cell with respect to the
        specified polygon. Default behavior compares each cell to the regular
        k-gon and does not weight the distances by cell area.

    """
    
    # This line is very "pythonic".
    weighted = True if areas is not None else False
    
    poly = Polygon()
    comp_poly = poly.regpoly(n) if ((n != -1) and (n != -2)) else None
    
    
    turn_dists = []
    #for the circle case, we have some work to do
    if n == -2:
        for cell in cells:
            
            polygon = cells[cell]
            vertices = []
            
            for i in range(len(polygon)):
                vertices.append(np.array(pos[polygon[i]]))
                
            
            disto = [0]+[  math.dist(vertices[i], vertices[i+1])  for i in range(len(vertices)-1) ] + [  math.dist(vertices[-1], vertices[0]) ]
            p = np.cumsum(disto)/sum(disto)




            pt = [0]+[cca( vertices[i], vertices[i+1], vertices[i+2]) for i in range(len(vertices)-2)] + [cca( vertices[-2], vertices[-1], vertices[0])] +  [cca( vertices[-1], vertices[0], vertices[1])]

            theta = np.cumsum(pt)
            dist = circ_dist(p,theta)
            if weighted:
                # if math.isnan(dist):
                #     print(areas[cell])
                #     print('hey!')
                turn_dists.append(dist* areas[cell])
            else:
                turn_dists.append(dist)
        
    
    #this needs to be expanded to include circle distances
    
    
    
    

    if n != -2:
        for cell in cells:
            
            polygon = cells[cell]
            vertices = []
            
            for i in range(len(polygon)):
                vertices.append(np.array(pos[polygon[i]]))
                
            if has_overlapping_vertices(vertices):
                turn_dists.append(0)
            else:
                if n == -1:
                    comp_poly = poly.regpoly(len(polygon))
                dist, _, _, _ = turning_function.distance(
                    vertices, 
                    comp_poly, 
                    brute_force_updates=False
                )
                if weighted:
                    turn_dists.append(dist * areas[cell])
                else:
                    turn_dists.append(dist)

    return turn_dists
