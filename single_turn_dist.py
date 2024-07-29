import turning_function
import numpy as np

from polygon import Polygon

def has_overlapping_vertices(vertices):
    """
    Returns whether the provided list of vertices contains two vertices located
    at the same position.

    Parameters
    ----------
    vertices : list
        List of np.array-like objects containing the (x,y) coordinates of each
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
                return True
    return False

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