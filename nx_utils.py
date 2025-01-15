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


#2-norm

def euc2(p, q):
    return  np.sqrt(    (p[0]- q[0])**2 + (p[1]- q[1])**2     )
    


#pluck function

def pluck(S):
    tol = 10**(-5)
    #last repeated index or current if not repeated
    repo = 0
    indo = [0]
    for i in range(len(S)):
        #is current not close to repo? If not, append
        if euc2(S[i], S[repo]) > tol:
            repo = i
            indo.append(i)
    return [S[i] for i in indo]   




def turning_distance(n, k):
    return np.pi*np.sqrt((4/(n*k))*  np.sum([(1/k*np.floor(i/n)-1/n*np.floor(i/k))**2 for i in range(k*n+1)])-(1/n-1/k)**2)


print(turning_distance(2,6))



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
    #for the circle case, incorporate circle distance
    if n == -2:
        for cell in cells:
            
            polygon = cells[cell]
            

            
            
            vertices = []
            
            for i in range(len(polygon)):
                vertices.append(np.array(pos[polygon[i]]))
                
                
            #pluck!
            vertices = pluck(vertices)
            if len(vertices)== 1:
                print('plucked!')
                

            disto = [0]+[  math.dist(vertices[i], vertices[i+1])  for i in range(len(vertices)-1) ] + [  math.dist(vertices[-1], vertices[0]) ]
            p = np.cumsum(disto)/sum(disto)




            pt = [0]+[cca( vertices[i], vertices[i+1], vertices[i+2]) for i in range(len(vertices)-2)] + [cca( vertices[-2], vertices[-1], vertices[0])] +  [cca( vertices[-1], vertices[0], vertices[1])]

            theta = np.cumsum(pt)
            dist = circ_dist(p,theta)
            if weighted:
                
                #For shard
                if len(vertices) == 2:
                    turn_dists.append( (np.pi/np.sqrt(12))      * areas[cell])
                elif len(vertices) == 1:
                    turn_dists.append( (np.pi/(len(polygon)*np.sqrt(3))   * areas[cell]))
                    
                # if math.isnan(dist):
                #     print(areas[cell])
                #     print('hey!')
                else:
                    turn_dists.append(dist* areas[cell])
            else:
                
                #For shard
                if len(vertices) == 2:
                    turn_dists.append( (np.pi/np.sqrt(12)))
                elif len(vertices) == 1:
                    turn_dists.append( (np.pi/(len(polygon)*np.sqrt(3))) )
                else:
                    turn_dists.append(dist)
        
    
    #this needs to be expanded to include circle distances
    
    
    
    

    if n != -2:
        for cell in cells:
            
            polygon = cells[cell]
            vertices = []
            
            for i in range(len(polygon)):
                vertices.append(np.array(pos[polygon[i]]))
                
            #pluck!
            vertices = pluck(vertices)
            
            #comp poly case
            if n == -1:
                comp_poly = poly.regpoly(len(polygon))
                if weighted:
                    
                    if len(vertices) == 2:
                        dist = turning_distance(2,len(polygon))
                        turn_dists.append( dist    * areas[cell])
                    elif len(vertices) == 1:
                        turn_dists.append( 0)
                    else:
                        dist, _, _, _ = turning_function.distance(
                            vertices, 
                            comp_poly, 
                            brute_force_updates=False
                        )
                        turn_dists.append( dist    * areas[cell])
                else:
                    
                    if len(vertices) == 2:
                        dist = turning_distance(2,len(polygon))
                        turn_dists.append( dist   )
                    elif len(vertices) == 1:
                        turn_dists.append( 0)
                    else:
                        dist, _, _, _ = turning_function.distance(
                            vertices, 
                            comp_poly, 
                            brute_force_updates=False
                        )
                        turn_dists.append( dist  )       
                    
            
            #finally, the six-sided case
            else:
                comp_poly = poly.regpoly(6)
                if weighted:
                    
                    if len(vertices) == 2:
                        dist = turning_distance(2,6)
                        turn_dists.append( dist    * areas[cell])
                    elif len(vertices) == 1:
                        dist = turning_distance(len(polygon),6)
                        turn_dists.append( dist    * areas[cell])
                    else:
                        dist, _, _, _ = turning_function.distance(
                            vertices, 
                            comp_poly, 
                            brute_force_updates=False
                        )
                        turn_dists.append( dist    * areas[cell])
                else:
                    
                    if len(vertices) == 2:
                        dist = turning_distance(2,6)
                        turn_dists.append( dist   )
                    elif len(vertices) == 1:
                        dist = turning_distance(len(polygon),6)
                        turn_dists.append( dist )
                    else:
                        dist, _, _, _ = turning_function.distance(
                            vertices, 
                            comp_poly, 
                            brute_force_updates=False
                        )
                        turn_dists.append( dist  )  
                
            

    return turn_dists
