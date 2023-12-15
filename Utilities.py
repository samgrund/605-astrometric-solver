import math
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
import astropy.units as u


# Parameters
from dotenv import load_dotenv
import os
load_dotenv()
PIXEL_SCALE = float(os.getenv("PIXEL_SCALE")) # [deg/px]

def orientation(p, q, r):
    """
    Determines the orientation of three points (p, q, r).
    Returns 0 if they are collinear, 1 if clockwise, and -1 if counterclockwise.
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else -1

def graham_scan(points):
    """
    Sorts a list of polygon nodes using the Graham's Scan algorithm
    to compute the convex hull of the polygon.
    Returns the sorted list of nodes forming the convex hull.
    """
    if len(points) < 3:
        return points  # Convex hull not possible with less than 3 points

    # Find the point with the lowest y-coordinate (and leftmost if tied)
    pivot = min(points, key=lambda point: (point[1], point[0]))

    # Sort the points based on the polar angle with respect to the pivot point
    sorted_points = sorted(points, key=lambda point: (math.atan2(point[1] - pivot[1], point[0] - pivot[0]), point))

    # Initialize the convex hull with the first two points
    convex_hull = [pivot, sorted_points[0]]

    # Traverse the sorted points and build the convex hull
    for i in range(1, len(sorted_points)):
        while len(convex_hull) > 1 and orientation(convex_hull[-2], convex_hull[-1], sorted_points[i]) != -1:
            convex_hull.pop()
        convex_hull.append(sorted_points[i])

    return convex_hull

def generate_high_visibility_colors(num_colors):
    # Generate an array of evenly spaced numbers
    nums = np.linspace(0, 1, num_colors)

    # Create a color map using these numbers
    colors = plt.cm.get_cmap('hsv')(nums)

    return colors

def create_wcs(local_asterism,gaia_asterism):
    local_xs = np.array([local_asterism['xa'],local_asterism['xc'], local_asterism['xd'], local_asterism['xb']])
    local_ys = np.array([local_asterism['ya'],local_asterism['yc'], local_asterism['yd'], local_asterism['yb']])

    gaia_xys = np.array([(gaia_asterism['xa'],gaia_asterism['ya']),(gaia_asterism['xc'],gaia_asterism['yc']),(gaia_asterism['xd'],gaia_asterism['yd']),(gaia_asterism['xb'],gaia_asterism['yb'])])
    gaia_skycoords = SkyCoord(ra=gaia_xys[:,0]*u.deg,dec=gaia_xys[:,1]*u.deg)
    wcs = fit_wcs_from_points((local_xs,local_ys),gaia_skycoords)
    return wcs

def local_asterism_to_coords(asterism):
    """
    Converts a local asterism a QTable containing the x and y coordinates of the stars. (In image coordinate system)
    """
    table = QTable(names=('x', 'y'), dtype=('float', 'float'))
    
    x1,y1 = asterism['xa'],asterism['ya']
    x2,y2 = asterism['xb'],asterism['yb']
    x3,y3 = asterism['xc'],asterism['yc']
    x4,y4 = asterism['xd'],asterism['yd']
    
    # Add the stars to the table
    table.add_row([x1,y1])
    table.add_row([x2,y2])
    table.add_row([x3,y3])
    table.add_row([x4,y4])
    
    return table

def gaia_asterism_to_coords(asterism):
    """
    Converts an asterism to a QTAble containing the RA and DEC coordinates of the stars.
    """
    table = QTable(names=('ra', 'dec'), dtype=('float', 'float'))

    ra1,dec1 = asterism['xa'],asterism['ya']
    ra2,dec2 = asterism['xb'],asterism['yb']
    ra3,dec3 = asterism['xc'],asterism['yc']
    ra4,dec4 = asterism['xd'],asterism['yd']
    
    table.add_row([ra1,dec1])
    table.add_row([ra2,dec2])
    table.add_row([ra3,dec3])
    table.add_row([ra4,dec4])
    
    return table
