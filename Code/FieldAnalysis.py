from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

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
    wcs = _get_wcs(hdul)
    imageshape = hdul[0].data.shape
    pixelarea = wcs.pixel_scale_matrix[1,1] * wcs.pixel_scale_matrix[0,0]
    imagearea = pixelarea * imageshape[0] * imageshape[1]
    return imagearea

def get_pixel_scale(hdul):
    return _get_wcs(hdul).pixel_scale_matrix[1,1]

def deg_to_px(hdul,deg):
    return deg / get_pixel_scale(hdul)

def get_diagonal_distance(hdul):
    """
    Gets the diagonal distance of the image in degrees.
    """
    wcs = _get_wcs(hdul)
    diagonal_px = np.sqrt(hdul[0].data.shape[0]**2 + hdul[0].data.shape[1]**2)
    diagonal_phys = diagonal_px * wcs.pixel_scale_matrix[1,1] # Assuming square pixels for simplicity
    return diagonal_phys
    