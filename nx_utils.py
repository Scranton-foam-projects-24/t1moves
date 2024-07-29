import turning_function
import statistics as stat
import numpy as np

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
            if vertex1[0] == vertex2[0] and vertex1[1] == vertex2[1] and i != j:
                print("CONTAINS DUPLICATES")
                return True
    return False

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
    # Try writing the output to a file and seeing what pos is
    print("pos:", pos)
    poly = Polygon()
    comp_poly = poly.regpoly(n) if n != -1 else None
    
    turn_dists = []
    
    for cell in cells:
        polygon = cells[cell]
        # print(len(polygon))
        vertices = []
        comp_len = len(polygon)
        for i in range(comp_len):
            print("adding:", np.array(pos[polygon[i]]))
            vertices.append(np.array(pos[polygon[i]]))
            
        print(vertices)
            
        if has_overlapping_vertices(vertices):
            continue
            
        if n == -1:
            comp_poly = poly.regpoly(comp_len)
        # print("calling turning_function.distance()")
        dist, _, _, _ = turning_function.distance(
            vertices, 
            comp_poly, 
            brute_force_updates=False
        )
        # print("computed turning distance")
        turn_dists.append(dist)

    return stat.mean(turn_dists)

if __name__ == "__main__":
    poly = Polygon()

    points = [np.array([0.57608955, 0.5807441 ]), np.array([0.57608955, 0.5807441 ]), np.array([0.57608955, 0.5807441 ])]

    print(has_overlapping_vertices(points))

    comp_poly = poly.regpoly(len(points))
    
    # dist, _, _, _ = turning_function.distance(
    #     points, 
    #     comp_poly, 
    #     brute_force_updates=False
    # )

    poly.graphpoly(points)
    poly.graphpoly(comp_poly)

    # print(dist)
        
    