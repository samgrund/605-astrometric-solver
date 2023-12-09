import numpy as np
from astropy.table import QTable
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Local modules
from StarDetection import extract_stars
from FieldAnalysis import get_image_area_physical, get_pixel_scale, deg_to_px
from Visualization import plot_stellar
from Utilities import graham_scan
from Utilities import generate_high_visibility_colors


# Parameters
from dotenv import load_dotenv
import os
load_dotenv()
ASTERISM_SIZE = int(os.getenv("ASTERISM_SIZE"))
FIELD_DENSITY = float(os.getenv("FIELD_DENSITY"))
PIXEL_SCALE = float(os.getenv("PIXEL_SCALE")) # [deg/px]

def _get_brightest_stars(hdul):
    """
    Gets the appropriate number of brightest stars from the image according to the field density.
    """
    # Extract stars
    imagedata = hdul[0].data
    stars = extract_stars(imagedata)
    
    # Get the number of stars
    n_stars = int(FIELD_DENSITY * get_image_area_physical(hdul))
    
    # Sort the stars by brightness and return the brightest ones
    stars.sort('flux',reverse=True)
    return stars[:n_stars]

def _get_asterisms_GAIA(stars):
    """
    Returns a QTable with 8 rows and 4 columns containing the coordinates of the stars in the asterism.
    """
    ASTERISM_RMIN = float(os.getenv("ASTERISM_RMIN_DEG"))
    ASTERISM_RMAX = float(os.getenv("ASTERISM_RMAX_DEG"))
    
    table = QTable(names=('xa','ya','xc','yc','xd','yd','xb','yb','rc','rd','rb'),dtype=('float','float','float','float','float','float','float','float', 'float','float','float'))
    stars_original = stars.copy()
    
    for star in stars:
        stars = stars_original.copy()
        # Remove the current star from the list of stars to avoid choosing it again
        stars.remove_row(np.where(stars['ra'] == star['ra'])[0][0])
        
        # First criteria is that the furthest star must be within the radius window
        distances = np.sqrt((star['ra'] - stars['ra'])**2 + (star['dec'] - stars['dec'])**2)
        relevant_stars = stars[(distances > ASTERISM_RMIN) & (distances < ASTERISM_RMAX)]
        relevant_stars.sort('phot_g_mean_flux',reverse=True)
        chosen_star = relevant_stars[0] # Brightest star within the radius window
        # Remove the chosen star from the list of stars to avoid choosing it again
        stars.remove_row(np.where(stars['ra'] == chosen_star['ra'])[0][0])
        
        # Second criteria is that the 2 remaining stars must be within a radius of 1/2 of the distance between the first and furthest star, measured from the midpoint between the first and furthest star
        distances = np.sqrt((star['ra'] - stars['ra'])**2 + (star['dec'] - stars['dec'])**2)
        distance_furthest = np.sqrt((star['ra'] - chosen_star['ra'])**2 + (star['dec'] - chosen_star['dec'])**2)
        midpoint = (star['ra'] + chosen_star['ra'])/2, (star['dec'] + chosen_star['dec'])/2
        distances_midpoint = np.sqrt((midpoint[0] - stars['ra'])**2 + (midpoint[1] - stars['dec'])**2)
        # Find stars that are within 1/2*distance_furthest from the midpoint
        relevant_stars = stars[(distances_midpoint < (distance_furthest/2)) & (distances < distance_furthest)]
        if len(relevant_stars) < 2:
            continue
        else:
            relevant_stars.sort('phot_g_mean_flux',reverse=True)
            chosen_stars = relevant_stars[:2]
            distances_chosen = np.sqrt((star['ra'] - chosen_stars['ra'])**2 + (star['dec'] - chosen_stars['dec'])**2)
            chosen_stars.add_column(distances_chosen,name='distances')
            chosen_stars.sort('distances')
            table.add_row([star['ra'],star['dec'],chosen_stars['ra'][0],chosen_stars['dec'][0],chosen_stars['ra'][1],chosen_stars['dec'][1],chosen_star['ra'],chosen_star['dec'],chosen_stars['distances'][0],chosen_stars['distances'][1],distance_furthest])
    return table

def _get_asterisms_image(stars):
    """
    Returns a QTable with 8 rows and 4 columns containing the coordinates of the stars in the asterism.
    """    
    ASTERISM_RMIN = float(os.getenv("ASTERISM_RMIN_PX"))
    ASTERISM_RMAX = float(os.getenv("ASTERISM_RMAX_PX"))
    
    table = QTable(names=('xa','ya','xc','yc','xd','yd','xb','yb','rc','rd','rb'),dtype=('float','float','float','float','float','float','float','float', 'float','float','float'))
    stars_original = stars.copy()
    
    for star in stars:
        stars = stars_original.copy()
        # Remove the current star from the list of stars to avoid choosing it again
        stars.remove_row(np.where(stars['x'] == star['x'])[0][0])
        
        # First criteria is that the furthest star must be within the radius window
        distances = np.sqrt((star['x'] - stars['x'])**2 + (star['y'] - stars['y'])**2)
        relevant_stars = stars[(distances > ASTERISM_RMIN) & (distances < ASTERISM_RMAX)]
        relevant_stars.sort('flux',reverse=True)
        chosen_star = relevant_stars[0] # Brightest star within the radius window
        # Remove the chosen star from the list of stars to avoid choosing it again
        stars.remove_row(np.where(stars['x'] == chosen_star['x'])[0][0])
        
        # Second criteria is that the 2 remaining stars must be within a radius of 1/2 of the distance between the first and furthest star, measured from the midpoint between the first and furthest star
        distances = np.sqrt((star['x'] - stars['x'])**2 + (star['y'] - stars['y'])**2)
        distance_furthest = np.sqrt((star['x'] - chosen_star['x'])**2 + (star['y'] - chosen_star['y'])**2)
        midpoint = (star['x'] + chosen_star['x'])/2, (star['y'] + chosen_star['y'])/2
        distances_midpoint = np.sqrt((midpoint[0] - stars['x'])**2 + (midpoint[1] - stars['y'])**2)
        # Find stars that are within 1/2*distance_furthest from the midpoint
        relevant_stars = stars[(distances_midpoint < (distance_furthest/2)) & (distances < distance_furthest)]
        if len(relevant_stars) < 2:
            continue
        else:
            relevant_stars.sort('flux',reverse=True)
            chosen_stars = relevant_stars[:2]
            distances_chosen = np.sqrt((star['x'] - chosen_stars['x'])**2 + (star['y'] - chosen_stars['y'])**2)
            chosen_stars.add_column(distances_chosen,name='distances')
            chosen_stars.sort('distances')
            table.add_row([star['x'],star['y'],chosen_stars['x'][0],chosen_stars['y'][0],chosen_stars['x'][1],chosen_stars['y'][1],chosen_star['x'],chosen_star['y'],chosen_stars['distances'][0],chosen_stars['distances'][1],distance_furthest])

    return table

def _asterisms_to_geometric_hash(asterisms,coords='image'):
    """
    Converts asterisms to a geometric hash.
    Returns a QTable where each row is an asterism and the columns are the geometric hash components.
    """
    # Define output table
    table = QTable(names=('xa','ya','xc','yc','xd','yd','xb','yb','rc','rd','rb'),dtype=('float','float','float','float','float','float','float','float', 'float','float','float'))
    asterism_output_table = QTable(names=('xa','ya','xc','yc','xd','yd','xb','yb','rc','rd','rb'),dtype=('float','float','float','float','float','float','float','float', 'float','float','float'))
    # Let's do some vector math
    for asterism in asterisms:
            
        AB = np.array([asterism['xb'] - asterism['xa'],asterism['yb'] - asterism['ya']])
        AC = np.array([asterism['xc'] - asterism['xa'],asterism['yc'] - asterism['ya']])
        AD = np.array([asterism['xd'] - asterism['xa'],asterism['yd'] - asterism['ya']])
        AA = np.array([asterism['xa'] - asterism['xa'],asterism['ya'] - asterism['ya']])
        
        if coords == 'image':
            AB *= PIXEL_SCALE
            AC *= PIXEL_SCALE
            AD *= PIXEL_SCALE
            AA *= PIXEL_SCALE
            
        AB_length = np.linalg.norm(AB)
        
        # Rotate AB 45 degrees clockwise to form x axis
        x = np.array([AB[0]*np.cos(np.pi/4) - AB[1]*np.sin(np.pi/4),AB[0]*np.sin(np.pi/4) + AB[1]*np.cos(np.pi/4)])/np.sqrt(2)
        
        # Rotate AB 45 degrees counter-clockwise to form y axis
        y = np.array([AB[0]*np.cos(-np.pi/4) - AB[1]*np.sin(-np.pi/4),AB[0]*np.sin(-np.pi/4) + AB[1]*np.cos(-np.pi/4)])/np.sqrt(2)
        
        # Project the stars onto the x and y axis
        xa = np.dot(AA,x)/np.linalg.norm(x)
        ya = np.dot(AA,y)/np.linalg.norm(y)
        xb = np.dot(AB,x)/np.linalg.norm(x)
        yb = np.dot(AB,y)/np.linalg.norm(y)
        xc = np.dot(AC,x)/np.linalg.norm(x)
        yc = np.dot(AC,y)/np.linalg.norm(y)
        xd = np.dot(AD,x)/np.linalg.norm(x)
        yd = np.dot(AD,y)/np.linalg.norm(y)
        
        # Calculate the distances between the stars
        rc = np.linalg.norm(np.array([xc,yc]))
        rd = np.linalg.norm(np.array([xd,yd]))
        rb = np.linalg.norm(np.array([xb,yb]))

        # New, non-normalized version
        #if (xc > AB_length or yc > AB_length or xd > AB_length or yd > AB_length) or (xc < 0 or yc < 0 or xd < 0 or yd < 0):
        #    continue
        #else:
            #print(f"Found asterism of radius {AB_length:.2e} [deg]")
        #    table.add_row([xa,ya,xc,yc,xd,yd,xb,yb,rc, rd, rb])
        #    asterism_output_table.add_row(asterism)
        
        table.add_row([xa,ya,xc,yc,xd,yd,xb,yb,rc, rd, rb])
        asterism_output_table.add_row(asterism)
    
    return table, asterism_output_table
    
def plot_asterisms_GAIA(asterisms,stars):
    """
    Visualize the asterisms on a background of GAIA stars.
    """
    # Turn asterisms into a lists of tuples
    coords_list = []
    for asterism in asterisms:
        coords_list.append([(asterism['xa'],asterism['ya']),(asterism['xc'],asterism['yc']),(asterism['xd'],asterism['yd']),(asterism['xb'],asterism['yb'])])
        
    sorted_coords = []
    for coords in coords_list:
        sorted_coords.append(graham_scan(coords))
        
    sizes = np.sqrt(stars['phot_g_mean_flux']/np.max(stars['phot_g_mean_flux']))*75
        
    fig,ax = plt.subplots()
    ax.scatter(stars['ra'],stars['dec'],marker='o',s=sizes,color='black')
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    
    num_asterisms = len(sorted_coords)
    generated_colors = generate_high_visibility_colors(num_asterisms)

    for (asterism,color) in zip(sorted_coords,generated_colors):
        ax.add_patch(plt.Polygon(asterism,closed=True,fill=False,color=color,linewidth=1.5, alpha=.8))

def plot_asterisms(hdul,asterisms,stars):
    """
    Visualize the asterisms on the image.
    """
    # Turn asterisms into a lists of tuples
    coords_list = []
    for asterism in asterisms:
        coords_list.append([(asterism['xa'],asterism['ya']),(asterism['xc'],asterism['yc']),(asterism['xd'],asterism['yd']),(asterism['xb'],asterism['yb'])])
        
    sorted_coords = []
    for coords in coords_list:
        sorted_coords.append(graham_scan(coords))
    
    fig,ax = plot_stellar(hdul[0].data,vlims=[5,99.5],cmap='gray_r')
    ax.scatter(stars['x'],stars['y'],marker='o',s=30,facecolors='none',edgecolors='white')
    # Draw each asterism as a polygon with its own color
    
    num_asterisms = len(sorted_coords)
    generated_colors = generate_high_visibility_colors(num_asterisms)
    
    for (asterism,color) in zip(sorted_coords,generated_colors):
        ax.add_patch(plt.Polygon(asterism,closed=True,fill=False,color=color,linewidth=1, alpha=0.5))
    
def _add_index_to_table(table):
    table.add_column(np.arange(len(table)),name='index')
    table.add_index('index')
    return table

def build_asterisms_from_GAIA(stars,visualize=False):
    """
    Builds asterisms from GAIA stars around a coordinate.
    """
    asterisms = _get_asterisms_GAIA(stars)
    hashes,asterisms = _asterisms_to_geometric_hash(asterisms,coords='radec')
    hashes, asterisms = _add_index_to_table(hashes), _add_index_to_table(asterisms)
    if visualize:
        plot_asterisms_GAIA(asterisms,stars)
    return asterisms,hashes

def build_asterisms_from_input_image(hdul,visualize=False):
    """
    Builds asterisms from an input image.
    """
    stars = _get_brightest_stars(hdul)
    asterisms = _get_asterisms_image(stars)
    hashes,asterisms = _asterisms_to_geometric_hash(asterisms,coords='image')
    hashes, asterisms = _add_index_to_table(hashes), _add_index_to_table(asterisms)
    if visualize:
        plot_asterisms(hdul,asterisms,stars)
    return asterisms,hashes    