import numpy as np
from astropy.table import QTable
import matplotlib.pyplot as plt

# Local modules
from AsterismBuilder import build_asterisms_from_GAIA
from AsterismBuilder import build_asterisms_from_input_image
from IndexHandler import get_index
from Utilities import create_wcs


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

def _test_alignment(hdul,hashnormdiffs,asterisms_GAIA,asterisms_LOCAL,hashes_GAIA,hashes_LOCAL):
    """
    Tests the alignment of the hashes. Returns a QTable with the following columns:
        'id_GAIA' - The id of the GAIA asterism
        'id_LOCAL' - The id of the LOCAL asterism
        'norm_diff' - The norm of the difference between the hashes
    """
    
    proposed_match = hashnormdiffs[0] # The proposed match is the one with the smallest norm_diff
    GAIA_asterism  = asterisms_GAIA[proposed_match['id_GAIA']]
    LOCAL_asterism = asterisms_LOCAL[proposed_match['id_LOCAL']]
    GAIA_hash = hashes_GAIA[proposed_match['id_GAIA']]
    LOCAL_hash = hashes_LOCAL[proposed_match['id_LOCAL']]
    
    # Calculate the WCS for the proposed match    
    wcs = create_wcs(LOCAL_asterism,GAIA_asterism)
    
    return wcs,hashnormdiffs[0]['norm_diff'], GAIA_asterism, LOCAL_asterism, GAIA_hash, LOCAL_hash

def solve_field(hdul,ra_approx,dec_approx):
    """
    Main function for solving the field. Calls the rest of the subfunctions.
    Returns a wcs object.    
    """
    gaia_stars = get_index(hdul,ra_approx,dec_approx)
    asterisms_GAIA, hashes_GAIA = build_asterisms_from_GAIA(gaia_stars)
    asterisms_LOCAL, hashes_LOCAL = build_asterisms_from_input_image(hdul)
    hashnormdiffs = match_hashes(hashes_GAIA, hashes_LOCAL, asterisms_GAIA, asterisms_LOCAL)
    wcs,score,gaia_asterism, local_asterism,gaia_hash,local_hash = _test_alignment(hdul,hashnormdiffs,asterisms_GAIA,asterisms_LOCAL, hashes_GAIA, hashes_LOCAL)
    return wcs,score,gaia_asterism,local_asterism,gaia_hash,local_hash,hashnormdiffs
    
    