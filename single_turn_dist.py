import turning_function
import numpy as np

from polygon import Polygon

poly = Polygon()

points = [np.array([0.76525199, 0.46502785]), np.array([0.76510414, 0.46327818]), np.array([0.76943905, 0.47132443]), np.array([0.76670732, 0.46963468])]

comp_poly = poly.regpoly(len(points))

dist, _, _, _ = turning_function.distance(
    points, 
    comp_poly, 
    brute_force_updates=False
)

poly.graphpoly(points)
poly.graphpoly(comp_poly)

print(dist)