from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
PIXEL_SCALE = float(os.getenv("PIXEL_SCALE")) # [deg/px]

def _get_wcs(hdul):
    """
    Gets the WCS from a FITS file.
    """
    header = hdul[0].header
    wcs = WCS(header)
    return wcs

def get_image_area_physical(hdul):
    """
    Gets the physical area of the image in square degrees.
    """
    pixel_size = PIXEL_SCALE # [deg/px]
    imageshape = hdul[0].data.shape
    pixelarea = pixel_size**2
    imagearea = pixelarea * imageshape[0] * imageshape[1]
    return imagearea

def get_pixel_scale():
    return PIXEL_SCALE  # [deg/px]

def deg_to_px(hdul,deg):
    return deg / get_pixel_scale()

def get_diagonal_distance(hdul):
    """
    Gets the diagonal distance of the image in degrees.
    """
    diagonal_px = np.sqrt(hdul[0].data.shape[0]**2 + hdul[0].data.shape[1]**2)
    diagonal_phys = diagonal_px * PIXEL_SCALE # Assuming square pixels for simplicity
    return diagonal_phys
    