import numpy as np

# GAIA
from astroquery.gaia import Gaia
from astropy.io.votable import parse

import glob

# Local modules
from FieldAnalysis import get_diagonal_distance

# .env parameters
from dotenv import load_dotenv
import os
load_dotenv()
FIELD_DENSITY = float(os.getenv("FIELD_DENSITY"))
SEARCH_RADIUS_MULTIPLIER = float(os.getenv("SEARCH_RADIUS_MULTIPLIER"))
RADIAL_TOLERANCE = float(os.getenv("RADIAL_TOLERANCE"))


def _check_cache(ra,dec):
    """
    Checks if the given coordinates are in the cache.
    """
    cache_entries = glob.glob('./GAIACache/*.votable.gz')
    filenames = [os.path.basename(entry) for entry in cache_entries]
    ras = [float(filename.split('_')[0]) for filename in filenames]
    decs = [float(filename.split('_')[1]) for filename in filenames]
    
    potential_name = f"./GAIACache/{ra:.2f}_{dec:.2f}_FD{FIELD_DENSITY:.2f}_SRM{SEARCH_RADIUS_MULTIPLIER:.2f}.votable.gz"
    return os.path.isfile(potential_name)

def _relevant_cache(ra,dec):
    """
    Tries to find a relevant cache entry for the given coordinates.
    """
    cache_entries = glob.glob('./GAIACache/*.votable.gz')
    filenames = [os.path.basename(entry) for entry in cache_entries]
    ras = [float(filename.split('_')[0]) for filename in filenames]
    decs = [float(filename.split('_')[1]) for filename in filenames]
    
    for i in range(len(ras)):
        if np.sqrt((ras[i] - ra)**2 + (decs[i] - dec)**2) < RADIAL_TOLERANCE:
            return cache_entries[i]
    return None

def _load_cache(filename):
    results = parse(filename).get_first_table().to_table()
    return results

def _get_stars_GAIA(hdul,ra,dec):
    """
    Gets the stars from GAIA DR2 around the given coordinates.
    """
    search_radius = get_diagonal_distance(hdul) * SEARCH_RADIUS_MULTIPLIER
    search_area = search_radius**2 * np.pi
    n_stars = int(FIELD_DENSITY * search_area)
    
    potential_name = f"./GAIACache/{ra:.2f}_{dec:.2f}_FD{FIELD_DENSITY:.2f}_SRM{SEARCH_RADIUS_MULTIPLIER:.2f}.votable.gz"
    
    relevant_cache = _relevant_cache(ra,dec)
    
    if relevant_cache is not None:
        # Just load the cache
        print("Using cached GAIA results.")
        return _load_cache(relevant_cache)
    else:
        print("Querying GAIA DR2 ...",end='')
        query = """SELECT TOP {}
                source_id, ra, dec, phot_g_mean_flux
                FROM gaiadr2.gaia_source
                WHERE CONTAINS(POINT('ICRS', gaiadr2.gaia_source.ra, gaiadr2.gaia_source.dec),
                CIRCLE('ICRS', {}, {}, {}))=1
                ORDER BY phot_g_mean_mag ASC""".format(n_stars,ra, dec, search_radius)
        
        job = Gaia.launch_job_async(query,
                                dump_to_file=True,output_format='votable',
                                output_file=potential_name,verbose=False)
        results = job.get_results()
        print("Done.")
        return results
    
    
    #if _check_cache(ra,dec):
    #    print("Using cached GAIA results.")
    #    return _load_cache(potential_name)
    #else:
    #    print("Querying GAIA DR2 ...")
    #    # Build the query
    #    query = """SELECT TOP {}
    #            source_id, ra, dec, phot_g_mean_flux
    #            FROM gaiadr2.gaia_source
    #            WHERE CONTAINS(POINT('ICRS', gaiadr2.gaia_source.ra, gaiadr2.#gaia_source.dec),
    #            CIRCLE('ICRS', {}, {}, {}))=1
    #            ORDER BY phot_g_mean_mag ASC""".format(n_stars,ra, dec, #search_radius)
    #    
    #    job = Gaia.launch_job_async(query,
    #                            dump_to_file=True,output_format='votable',
    #                            output_file=potential_name,verbose=False)
    #    results = job.get_results()
    #    print("Done.")
    #    return results


def get_index(hdul,ra,dec):
    """
    Returns the index DB for the given coordinates.
    Queries GAIA DR2 around the given coordinates.
    """
    GAIA_stars = _get_stars_GAIA(hdul,ra,dec)
    return GAIA_stars
    