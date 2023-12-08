import math
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

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

from astropy.wcs import WCS
from astropy.io import fits
import numpy as np

def create_wcs(pixel_coords, sky_coords):
    # Calculate rotation angle
    delta_x = pixel_coords[1][0] - pixel_coords[0][0]
    delta_y = pixel_coords[1][1] - pixel_coords[0][1]
    delta_ra = sky_coords[1][0] - sky_coords[0][0]
    delta_dec = sky_coords[1][1] - sky_coords[0][1]

    rotation_angle = np.degrees(np.arctan2(delta_y * delta_ra - delta_x * delta_dec, delta_x * delta_ra + delta_y * delta_dec))

    # Calculate physical distance covered by the image along each axis
    delta_ra = abs(sky_coords[1][0] - sky_coords[0][0])
    delta_dec = abs(sky_coords[1][1] - sky_coords[0][1])

    # Calculate pixel scale (CDELT) along each axis
    delta_x = abs(pixel_coords[1][0] - pixel_coords[0][0])
    delta_y = abs(pixel_coords[1][1] - pixel_coords[0][1])

    cdelt_x = delta_ra / delta_x
    cdelt_y = delta_dec / delta_y

    # Create a FITS header
    hdr = fits.Header()

    # Add necessary WCS keywords based on the provided coordinates
    hdr['CTYPE1'] = 'RA---TAN'  # Coordinate type for axis 1 (example: Right Ascension)
    hdr['CTYPE2'] = 'DEC--TAN'  # Coordinate type for axis 2 (example: Declination)

    hdr['CRVAL1'] = sky_coords[0][0]  # Reference value for RA
    hdr['CRVAL2'] = sky_coords[0][1]  # Reference value for DEC

    hdr['CRPIX1'] = pixel_coords[0][0]  # Reference pixel for axis 1
    hdr['CRPIX2'] = pixel_coords[0][1]  # Reference pixel for axis 2

    hdr['CUNIT1'] = 'deg'  # Units for axis 1 (degrees)
    hdr['CUNIT2'] = 'deg'  # Units for axis 2 (degrees)

    hdr['CDELT1'] = cdelt_x  # Pixel scale for axis 1
    hdr['CDELT2'] = cdelt_y  # Pixel scale for axis 2

    # Include rotation angle information
    rotation_matrix = np.array([[np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
                                [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))]])


    hdr['PC1_1'] = rotation_matrix[0, 0]
    hdr['PC1_2'] = rotation_matrix[0, 1]
    hdr['PC2_1'] = rotation_matrix[1, 0]
    hdr['PC2_2'] = rotation_matrix[1, 1]

    # Create a WCS object using the FITS header
    wcs = WCS(hdr)
    
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
