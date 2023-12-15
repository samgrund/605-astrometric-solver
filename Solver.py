import numpy as np
from astropy.table import QTable
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

# Local modules
from AsterismBuilder import build_asterisms_from_GAIA
from AsterismBuilder import build_asterisms_from_input_image
from IndexHandler import get_index
from Utilities import create_wcs

from dotenv import load_dotenv
import os
load_dotenv()
STAR_STD = float(os.getenv("STAR_STD"))


def match_hashes(hashes_GAIA, hashes_LOCAL, asterisms_GAIA, asterisms_LOCAL):
    """
    Matches the hashes from GAIA and LOCAL, i.e. the input image.
    """
    table = QTable(names=('id_GAIA','id_LOCAL','norm_diff'),dtype=('i4','i4','f4'))
    
    for id_LOCAL,(hash_LOCAL,asterism_LOCAL) in enumerate(zip(hashes_LOCAL,asterisms_LOCAL)):
        for id_GAIA,(hash_GAIA,asterism_GAIA) in enumerate(zip(hashes_GAIA,asterisms_GAIA)):
            dxb = hash_GAIA['xb'] - hash_LOCAL['xb']
            dyb = hash_GAIA['yb'] - hash_LOCAL['yb']
            dxc = hash_GAIA['xc'] - hash_LOCAL['xc']
            dyc = hash_GAIA['yc'] - hash_LOCAL['yc']
            dxd = hash_GAIA['xd'] - hash_LOCAL['xd']
            dyd = hash_GAIA['yd'] - hash_LOCAL['yd']
            diff_norm = np.sqrt(dxb**2 + dyb**2 + dxc**2 + dyc**2 + dxd**2 + dyd**2)
            
            table.add_row([id_GAIA,id_LOCAL,diff_norm])
            
    table.sort('norm_diff',reverse=False)
    return table

def _test_alignment(hdul,hashnormdiffs,asterisms_GAIA,asterisms_LOCAL,hashes_GAIA,hashes_LOCAL,gaia_stars):
    """
    Tests the alignment of the hashes. Returns a QTable with the following columns:
        'id_GAIA' - The id of the GAIA asterism
        'id_LOCAL' - The id of the LOCAL asterism
        'norm_diff' - The norm of the difference between the hashes
    """
    
    for hashnormdiff in hashnormdiffs:
        proposed_match = hashnormdiffs[0]
        GAIA_asterism  = asterisms_GAIA[proposed_match['id_GAIA']]
        LOCAL_asterism = asterisms_LOCAL[proposed_match['id_LOCAL']]
        GAIA_hash = hashes_GAIA[proposed_match['id_GAIA']]
        LOCAL_hash = hashes_LOCAL[proposed_match['id_LOCAL']]
        
        # Calculate the WCS for the proposed match    
        wcs = create_wcs(LOCAL_asterism,GAIA_asterism)
        
        # Estimate the Bayes factor for the proposed match
        corners = [[0,0],[0,4096],[4096,4096],[4096,0]]
        corner_coords = [wcs.pixel_to_world_values(x,y) for (x,y) in corners]

        stars_pixels = wcs.world_to_pixel_values(gaia_stars['ra'],gaia_stars['dec'])
        
        # Find stars in frame
        stars_in_frame = np.where((stars_pixels[0] > 0) & (stars_pixels[0] < 4096) & (stars_pixels[1] > 0) & (stars_pixels[1] < 4096))[0]
        
        stars_in_frame_xs = stars_pixels[0][stars_in_frame]
        stars_in_frame_ys = stars_pixels[1][stars_in_frame]
        stars_in_frame_to_coords = np.array([stars_in_frame_xs,stars_in_frame_ys])
        
        # Manual OVERRIDE for testing
        stddev = STAR_STD
        
        # Create a new image by adding gaussians at the positions of the gaia stars with the same stddev as the image
        gaia_image = np.zeros((4096,4096))
        gaia_image[stars_in_frame_to_coords[1].astype(int),stars_in_frame_to_coords[0].astype(int)] = 1
        gaia_image = ndimage.gaussian_filter(gaia_image,stddev)
        
        # Flat_image
        flat_image = np.zeros((4096,4096))
        flat_image += np.sum(gaia_image)/(4096*4096)
        
        # Normalized input image
        norm_image = hdul[0].data/np.max(hdul[0].data)
        
        K = np.sum((norm_image*gaia_image)**2)/np.sum((norm_image*flat_image)**2)

        #if K > 5e6:
        if True:
            return wcs,hashnormdiffs[0]['norm_diff'], GAIA_asterism, LOCAL_asterism, GAIA_hash, LOCAL_hash,K,stars_pixels
    raise Exception("No match found.")
    
def solve_field(hdul,ra_approx,dec_approx):
    """
    Main function for solving the field. Calls the rest of the subfunctions.
    Returns a wcs object.    
    """
    gaia_stars = get_index(hdul,ra_approx,dec_approx)
    asterisms_GAIA, hashes_GAIA,_,_ = build_asterisms_from_GAIA(gaia_stars)
    asterisms_LOCAL, hashes_LOCAL, _,_ = build_asterisms_from_input_image(hdul)
    
    print("Matching hashes... ",end="")
    hashnormdiffs = match_hashes(hashes_GAIA, hashes_LOCAL, asterisms_GAIA, asterisms_LOCAL)
    wcs,score,gaia_asterism, local_asterism,gaia_hash,local_hash,K,star_xys = _test_alignment(hdul,hashnormdiffs,asterisms_GAIA,asterisms_LOCAL, hashes_GAIA, hashes_LOCAL,gaia_stars)
    print(f"Match found! Bayes factor: {K:.2e}")
    
    outputdict = {
        'wcs':wcs,
        'delta':score,
        'K':K,
        'gaia_asterism':gaia_asterism,
        'local_asterism':local_asterism,
        'gaia_hash':gaia_hash,
        'local_hash':local_hash,
        'star_coords':star_xys,
    }
    
    return outputdict
    
    