import matplotlib.pyplot as plt
import numpy as np

class Polygon():
    """
    Creates a polygon with an initial number of sides
    """    

    def regpoly(self, n):
        """
        Return list of verticies of a polygon of n sides with perimeter = 1.
        """
    
        if n <= 1:
            raise ValueError("Too few number of sides.")
            
        if n > 1024:
            raise ValueError("Too many sides for turning-function package.")
            
        # Calculate length of individual side for polygon with perimeter = 1
        side_len = 1/n
    
        # Radius of circumcircle, also vertex to centroid distance.
        radii = [((side_len/2)/(np.sin(np.pi/n)))] * n
        
        # Calculate angle at origin which separates each point
        angles = [((2 * np.pi)/n)] * n
    
        return self.poly(angles, radii)
    
    def graphpoly(self, points):
        """
        Produces a plot visualizing a polygon constructed by the provided
        coordinates.
        """
        
        # Unzip list of coordinate pairs into two lists
        xpoints = []
        ypoints = []

        for x,y in points:
            xpoints.append(x)
            ypoints.append(y)
        
        # Connect the last point back to the first point        
        xpoints.append(points[0][0])
        ypoints.append(points[0][1])
        
        plt.plot(xpoints, ypoints)
        plt.show()
        
    def randpoly(self, n):
        x_coords = [np.random.uniform(-1, 1) for _ in range(n)]
        y_coords = [np.random.uniform(-1, 1) for _ in range(n)]
        points = [list(a) for a in zip(x_coords, y_coords)]

        return points            
    
    def poly(self, angles, radii):
        points = []
        points.append((radii[0],0))
        
        # Rotate about the circumcircle (2/n)π radians and place another point
        for i, angle in enumerate(angles):
        
            # Start rotation from the last point plotted
            x = points[-1][0]
            y = points[-1][1]
            
            # Rotate a line connecting the last point and the origin about the origin
            x_prime = x * np.cos(angle) - y * np.sin(angle)
            y_prime = y * np.cos(angle) + x * np.sin(angle)
                        
            radii_ratio = radii[i]/radii[i-1]
            
            new_x = round(radii_ratio * x_prime, 16)
            new_y = round(radii_ratio * y_prime, 16)
    
            points.append((new_x, new_y))
        
        return points
    
    def normpoly(self, n, mu=0.0, sigma=0.01):
        initpoly = self.regpoly(n)[0:-1]
        
        points = []
        for x,y in initpoly:
            np.random.normal
            x += np.random.normal(mu, sigma)
            y += np.random.normal(mu, sigma)
            points.append((x,y))
        
        return points

if __name__ == "__main__":
    p = Polygon()
    r = p.regpoly(6)
    print(r)
    tile = [(1.0, 1.0),
            (0.5, 0.0),
            (0.0, 1.0),
            (0.0, 2.0),
            (0.5, 3.0),
            (1.0, 2.0)]
    p.graphpoly(tile)
    