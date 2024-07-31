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
        List of array-like objects containing the (x,y) coordinates of each
        vertex as values.

    Returns
    -------
    bool
        The result of comparing the number of unique x-values with the total
        number of x-values and the number of unique y-values with the total
        number of y-values. 

    """
    print("has_overlapping_vertices() called!")
    for i, vertex1 in enumerate(vertices):
        for j, vertex2 in enumerate(vertices):
            if vertex1[0] == vertex2[0] and vertex1[1] == vertex2[1] and i != j:
                return True
    return False

print("began execution")
poly = Polygon()
print("here")
# points = [np.array([0.07258301, 0.32230874]), np.array([0.07258301, 0.32230874]), np.array([0.07258301, 0.32230874])]
# points = [np.array([0.92992764, 0.24358688]), np.array([0.90657019, 0.22709718]), np.array([0.90657019, 0.22709718])]
# points = [np.array([0.11259959, 0.57197477]), np.array([0.11259959, 0.57197477]), np.array([0.11259959, 0.57197477])]
# points = [np.array([0.8052397 , 0.04203877]), np.array([0.80945053, 0.03633673]), np.array([0.8052397 , 0.04203877])]
points = [np.array([0.61018154, 0.94410252]), np.array([0.61018154, 0.94410252]), np.array([0.61018154, 0.94410252])]
# print("here")
# Things go haywire when two points match
# EVERYTHING breaks when three points match

# pyplot.py runs into an issue before polygon.py can import numpy...

print(has_overlapping_vertices(points))

comp_poly = poly.regpoly(len(points))

if not has_overlapping_vertices(points):
    dist, _, _, _ = turning_function.distance(
        points, 
        comp_poly, 
        brute_force_updates=False
    )

    poly.graphpoly(points)
    poly.graphpoly(comp_poly)
    
    print(dist)
else:
    print("appended 0")