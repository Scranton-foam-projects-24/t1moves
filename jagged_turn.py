#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:45:26 2025

@author: r92830873
"""



#This is using the jagged network for the purpose of the turning distance


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:37:03 2020

@author: r92830873
"""

#For making a figure showing popping process for Voronoi and lattice initial conditions






#Updated for 11/12/2020






# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import random
from nx_utils_jagged import *


# make up data points


#Updated for6/27/2020

np.random.seed([193840])


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        


        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def polygon_area(vertices):
    """
    Calculates the area of a polygon using the shoelace formula.

    Args:
        vertices: A list of tuples representing the vertices of the polygon, 
                  in the form (x, y).

    Returns:
        The area of the polygon.
    """

    n = len(vertices)
    area = 0.0

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Wrap around to the first vertex for the last calculation
        area += (x1 * y2 - x2 * y1)
        
    return abs(area) / 2.0



#originally 30k cells

#How many cells
N = int(1000)


# ## Code for generating triangle lattice
# #points = []
# #s = int(np.sqrt(N))
# #c = 1/s
# #A = [np.array([0,i/s]) for i in range(2,s-2)]
# #AA = A
# #points+= A
# #ss = int(np.sqrt(N/3))
# #for i in range(2,ss-2):
# #    AA+= [qq+ np.array([np.sqrt(3/N)*i,0]) for qq in A]
# #BB = [qq+[c/2, np.sqrt(3)*c/2] for qq in AA]
# #
# #        
# #points = AA+BB   






# # # For generating Voronoi diagram
# points = np.random.uniform(0, 1, size=(N,2))


# #For generating hexagonal lattice
# # points = []
# # spacing = 25
# # vec1 = [np.sqrt(3)/2, 1/2]
# # vec2 = [np.sqrt(3)/2, -1/2]
# # pt = [0,0]
# # for j in range(-spacing,spacing):
# #     for k in range(-spacing,spacing):
# #         pt = [.5+((j+k)*np.sqrt(3)/2)/spacing, .5+((j-k)/2)/spacing]
# #         if pt[0] >= 0 and pt[0] <= 1 and pt[1] >= 0 and pt[1] <= 1 :
# #             points.append(pt)
            
        



# # compute Voronoi tesselation
# vor = Voronoi(points)

# # plot
# regions, vertices = voronoi_finite_polygons_2d(vor)



# #If you want a distribution of initial Voronoi diagram
# lenvec = [len(i) for i in regions]
# lenvechist = [lenvec.count(i) for i in range(2, 20)]
# plt.plot(lenvechist)

# min_x = 0
# max_x = 1
# min_y = 0
# max_y = 1

# mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
# bounded_vertices = np.max((vertices, mins), axis=0)
# maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
# bounded_vertices = np.min((bounded_vertices, maxs), axis=0)



# box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

# # colorize
# q = 0
# polys = []
# goodvertices = []
# polyind = [[] for i in range(len(regions))]
# p = 0
# edgeind = []

# # #If you want a pretty picture of initial conditions, use this
# # for region in regions:    
# #     polygon = vertices[region]
# #     # Clipping polygon
# #     poly = Polygon(polygon)
# #     poly = poly.intersection(box)
# #     polygon = [p for p in poly.exterior.coords]
# #     for v in polygon[0:-1]:
# #         if v in goodvertices:
# #             polyind[p].append(goodvertices.index(v))
# #         else:
# #             goodvertices.append(v)
# #             polyind[p].append(q)
# #             q+= 1
# #         plt.fill(*zip(*polygon), alpha=0.4)
# #     p+= 1

# # for pol in polyind:
# #     for i in range(len(pol)):
# #         e = {pol[i], pol[np.mod(i+1, len(pol))]} 
# #         if e not in edgeind:
# #             edgeind.append(e)

# # ledgeind = [list(i) for i in edgeind]
# # for e in ledgeind:
# #     plt.plot([goodvertices[e[0]][0], goodvertices[e[1]][0]], [goodvertices[e[0]][1], goodvertices[e[1]][1]], 'k-')
# # plt.xlim(min_x-.01, max_x+.01)
# # plt.ylim(min_y-.01, max_y+.01)

# #plt.savefig('voro2000.png')

# # plt.show()
            
# #If you don't care about graphing initial conditions, use this

# for region in regions:   
#     polygon = vertices[region]
#     # Clipping polygon
#     poly = Polygon(polygon)
#     poly = poly.intersection(box)
#     polygon = [p for p in poly.exterior.coords]
#     for v in polygon[0:-1]:
#         if v in goodvertices:
#             polyind[p].append(goodvertices.index(v))
#         else:
#             goodvertices.append(v)
#             polyind[p].append(q)
#             q+= 1
#     p+= 1

# for pol in polyind:
#     for i in range(len(pol)):
#         e = {pol[i], pol[np.mod(i+1, len(pol))]} 
#         if e not in edgeind:
#             edgeind.append(e)

# #print(polyind)
# #print(goodvertices)

# print('flag!')

# vertices = goodvertices







# #Let's put things back in language for normal Voronoi diagrams

# polys = [i for i in polyind]
# edges = [i for i in edgeind]



# #Given a cell containing an edge, what kind of neighbor is it?



# ####Some test patterns for grains:
    
# #Here's a moat scenario    
    
# #Moat grain
# A = [7,1,2,3,4,5,6,2,1,8,9,10,11, 12, 13, 14]

# #Island grain
# B = [2,3,4,5,6]

# #Grain at shore

# C = [1,7,15,16,17,8]


# ##Typical scenario


# #Edge neighbors
# A = [1,2,3,4,5,6]


# B = [2,1,10,11,12,13]

# #Vertex neighbors

# C = [2,3,16,15,14,13]

# D = [1,6,7,8,9,10]





# #Island scenario

# A = [1,7,6,2]

# B = [4,5,6,2,3]

# C = [10, 1, 7, 8, 9]



# ##Testing for normal break
# #A = [1,2,3,4,5,6]
# #B = [2,1,10,11,12,13]
# #C = [2,3,16,15,14,13]
# #D = [1,6,7,8,9,10]
# #Q = [A,B,C,D]
# #edge = [1,2]


# ##Testing for wall break
# #A = [1,7,6,2]
# #B = [4,5,6,2,3]
# #C = [10, 1, 7, 8, 9]
# #Q = [A,B,C]
# #edge = [1,2]

# #Testing for moat

# ##Moat grain
# #A = [7,1,2,3,4,5,6,2,1,8,9,10,11, 12, 13, 14]
# #B = [2,3,4,5,6]
# #C = [1,7,15,16,17,8]
# #Q = [A,B,C]
# #edge = [1,2]

  







#Code for 50 experiments of 30k to 3k popping                                                                                                                                                                                                                                                                                                                                          


# #original params
# tests = 50
# veclen = 25
# vordists = np.zeros((tests, 100, veclen))
# vormax = np.zeros((tests, 100))


#teszt params
tests = 1
veclen = 25
vordists = np.zeros((tests, 100, veclen))
vormax = np.zeros((tests, 100))




#Initial dist voronoi.

sampno = 0



#storing the dataset of turning distances


turndata = np.zeros(shape = (tests,6,11))



for qqq in range(tests):
    print(qqq)
    sampno = 0
    
    
    
    
    
    # # # For generating Voronoi diagram
    # points = np.random.uniform(0, 1, size=(N,2))
    
    
    # For generating hexagonal lattice
    #spacing was originally 161
    points = []
    spacing = 30
    vec1 = [np.sqrt(3)/2, 1/2]
    vec2 = [np.sqrt(3)/2, -1/2]
    pt = [0,0]
    for j in range(-spacing,spacing):
        for k in range(-spacing,spacing):
            pt = [.5+((j+k)*np.sqrt(3)/2)/spacing, .5+((j-k)/2)/spacing]
            if pt[0] >= 0 and pt[0] <= 1 and pt[1] >= 0 and pt[1] <= 1 :
                points.append(pt)
                
    print(len(points))
    
    
    # compute Voronoi tesselation
    vor = Voronoi(points)
    
    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    
    
    #If you want a distribution of initial Voronoi diagram
    lenvec = [len(i) for i in regions]
    vordists[qqq,0,:] = [lenvec.count(i) for i in range(25)]
    vormax[qqq,0] = max(lenvec)

    
    min_x = 0
    max_x = 1
    min_y = 0
    max_y = 1
    
    mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
    bounded_vertices = np.max((vertices, mins), axis=0)
    maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
    bounded_vertices = np.min((bounded_vertices, maxs), axis=0)
    
    
    
    box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
    
    # colorize
    q = 0
    polys = []
    goodvertices = []
    polyind = [[] for i in range(len(regions))]
    p = 0
    edgeind = []
    
    
    
    for region in regions:   
        polygon = vertices[region]
        # Clipping polygon
        poly = Polygon(polygon)
        poly = poly.intersection(box)
        polygon = [p for p in poly.exterior.coords]
        for v in polygon[0:-1]:
            if v in goodvertices:
                polyind[p].append(goodvertices.index(v))
            else:
                goodvertices.append(v)
                polyind[p].append(q)
                q+= 1
        p+= 1
        # print(p)

    for pol in polyind:
        for i in range(len(pol)):
            e = {pol[i], pol[np.mod(i+1, len(pol))]} 
            if e not in edgeind:
                edgeind.append(e)



    vertices = goodvertices

    polys = [i for i in polyind]
    edges = [i for i in edgeind]


    
    Qind = 0
    outfaceind = []
    Q = []
    outface = []
    inface = []
    noside = 0

    #Gotta have my pops...
    pops = 1
    sampno = 1
    
    
    vlist = [ [vertices[i] for i in polys[j]][::-1] for j in range(len(polys))]

    areas = [ polygon_area(vlist[j]) for j in range(len(polys))]
    
    turndata[qqq,0,0] = sum(network_disorder(vlist, n=-2))/len(areas)
    turndata[qqq,1,0] = sum(network_disorder(vlist, n=-2, areas = areas))
    turndata[qqq,2,0] = sum(network_disorder(vlist, n=6))/len(areas)
    turndata[qqq,3,0] = sum(network_disorder(vlist, n=6, areas = areas))
    turndata[qqq,4,0] = sum(network_disorder(vlist, n=-1))/len(areas)
    turndata[qqq,5,0] = sum(network_disorder(vlist, n=-1, areas = areas))

    
    
    
    #pops were at 27k
    
    while pops <901:
        outfaceind = []
        inface = []
    #    print(pops)
        #first we have to pick a random edge
        noside = 0
        edgewalls = 0
        while noside == 0:
            edge = list(np.random.choice(edges))
            #Check to see if vertex isn't on boundary (should be pretty cheap move)
            if ((0 not in vertices[edge[0]]) and (1 not in vertices[edge[0]])) or ((0 not in vertices[edge[1]]) and (1 not in vertices[edge[1]])):
                noside = 1
                if (0 in vertices[edge[0]]) or (0 in vertices[edge[1]]):
                    edgewalls +=1 
                if (1 in vertices[edge[0]]) or (1 in vertices[edge[1]]):
                    edgewalls += 1
    #    print(edge)
        Q = []
        #tags mean 0 for vertex neighbor, and 1 for edge neighbor
        Qtag = []
        esum = 0
        vmin = 10
        QEind = []
        QVind = []
        #What faces contain edge vertices?
        for i in range(len(polys)):
            if (edge[0] in polys[i]) and (edge[1] in polys[i]):
                outfaceind.append(i)
                Q.append(polys[i])
                Qtag.append(1)
                esum += len(polys[i])
                QEind.append(i)
            if (edge[0] in polys[i]) ^ (edge[1] in polys[i]):
                outfaceind.append(i)
                Q.append(polys[i])
                Qtag.append(0)
                vmin = min(vmin, len(polys[i]))
                QVind.append(i)
    
        QEind.reverse()        
    
    
    
    #We can check for whether this edge can pop
    
        if ((len(Q) == 4) or ((len(Q) == 3) and edgewalls == 1) or ((len(Q) == 3) and edgewalls == 2)) and vmin>3 and esum>7:
    
            
            for j in QVind:
                if edge[0] in polys[j]:
                    polys[j].remove(edge[0])
                else:
                    polys[j].remove(edge[1])
            
            #Vertex shedding from polys
            
    
    
            #Remove edge neighbors from poly list        
            for ind in QEind:
                del polys[ind]
    
            
            #Time to add the big grain if there are two edge neighbors            
            qq = np.where(np.array(Qtag) == 1)[0]
            A = np.array(Q[qq[0]])
            B = np.array(Q[qq[1]])
            ll = int(np.where(A == edge[0])[0])
            if A[(ll+1)%len(A)] == edge[1]:
                arc1 = [A[(k+2+ll)%len(A)] for k in range(len(A)-2)]
            else:
                arc1 = [A[(ll-2-k)%len(A)] for k in range(len(A)-2)]
                
                
            ll = int(np.where(B == edge[1])[0])
            if B[(ll+1)%len(B)] == edge[0]:
                arc2 = [B[(k+2+ll)%len(B)] for k in range(len(B)-2)]
            else:
                arc2 = [B[(ll-2-k)%len(B)] for k in range(len(B)-2)]
        
            
            polys.append(arc1+arc2)
    
            
    
        
            uneigh = []
            vneigh = []
            #This loop finds u and v neighbors, and also deletes all edges containing u or v
            for k in reversed(range(len(edges))):
                if edge[0] in edges[k]:
                    if edges[k] != set(edge):
    #                    print(edges[k])
                        uneigh.append(list(edges[k].difference({edge[0]}))[0])
                    del edges[k]
                    continue
            
                if edge[1] in edges[k]:
                    if edges[k] != set(edge):
    #                    print(edges[k])
                        vneigh.append(list(edges[k].difference({edge[1]}))[0])
                    del edges[k]
            edges.append(set(uneigh))
            edges.append(set(vneigh))
            
            
    
    
            #Here, we insert the turning distances
            if pops % 90 == 0:
                #Stats
                lenvec = [len(i) for i in polys]
                vordists[qqq, sampno,:] = [lenvec.count(i) for i in range(25)]
                vormax[qqq,sampno] = max(lenvec)
                
                vlist = [ [vertices[i] for i in polys[j]][::-1] for j in range(len(polys))]
                areas = [ polygon_area(vlist[j]) for j in range(len(polys))]
                turndata[qqq,0,sampno] = sum(network_disorder(vlist, n=-2))/len(areas)
                turndata[qqq,1,sampno] = sum(network_disorder(vlist, n=-2, areas = areas))
                turndata[qqq,2,sampno] = sum(network_disorder(vlist, n=6))/len(areas)
                turndata[qqq,3,sampno] = sum(network_disorder(vlist, n=6, areas = areas))
                turndata[qqq,4,sampno] = sum(network_disorder(vlist, n=-1))/len(areas)
                turndata[qqq,5,sampno] = sum(network_disorder(vlist, n=-1, areas = areas))
                sampno += 1
                


                
            pops += 1

            


#Save data

# np.save('vor30kmcdists.npy', vordists)
# np.save('vor30kgel.npy', vormax)    

# np.save('vorhex30kmcdists.npy', vordists)
# np.save('vorhex30kgel.npy', vormax)  




#Testing what dists look like

# for i in range(tests):
    
#     plt.plot(vormax[i,:])


for i in range(tests):
    
    plt.plot(vordists[i,:, 3])




#Some functions to collect stats

# #Frequencies
# def getfreqs(polys):
#     zz1 = [len(i) for i in polys]
#     zz2 = [sum( [i == j for i in zz1]) for j in range(3,1000)]
#     return [i*(1/len(zz1)) for i in zz2]

 
# #Weights


# plt.plot(getfreqs(polys)[0:15])
# #
# plt.plot(getfreqs(polys))

# #Max grain
# maxpoly= max([len(i) for i in polys])


#If you want a figure
ledgeind = [list(i) for i in edges]
for e in ledgeind:
    plt.plot([vertices[e[0]][0], vertices[e[1]][0]], [vertices[e[0]][1], vertices[e[1]][1]], 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1) 
plt.axis('equal')
plt.axis('off')
plt.show()



#So then you want to compute turning distance on this thing


#plotting the turning distances

plt.plot(turndata[qqq,0,:], label = "k-gon")
plt.plot(turndata[qqq,1,:], label = "k-gon, weighted")
plt.plot(turndata[qqq,2,:], label = '6-gon')
plt.plot(turndata[qqq,3,:], label = '6-gon, weighted')
plt.plot(turndata[qqq,4,:], label = 'circle')
plt.plot(turndata[qqq,5,:], label = 'circle, weighted')
plt.legend()
plt.show()





