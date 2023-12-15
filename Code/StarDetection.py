import numpy as np
from astropy.table import QTable
from astropy.convolution import convolve
from skimage import measure
from scipy.optimize import curve_fit
from scipy import integrate
from photutils import CircularAperture, CircularAnnulus, aperture_photometry

import warnings
warnings.filterwarnings("ignore") # Ignore warnings

# --- Settings ---
n_sigma = 3 # Number of sigma above background to consider a peak
window_size = 9 # Size of the window to use for the 2D Gaussian fit
n_sigma_fwhm = 3 # Number of sigma to use to filter out stars that are too large
apperture_radius_multiplier = 2.5 # Multiplier to use to calculate the aperture radius, relative to the FWHM
# ----------------

def _gaussian_2D(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian with rotation and offset.
    """
    x, y = xy
    x, y = x - xo, y - yo
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    return offset + amplitude * np.exp(-(a * x ** 2 + 2 * b * x * y + c * y ** 2))

def _fit_2D_gaussian(array):
    """
    Fits a 2D Gaussian to a 2D array.
    """
    arrayshape = array.shape

    # Create a meshgrid for coordinates
    x, y = np.meshgrid(np.arange(arrayshape[1]), np.arange(arrayshape[0]))

    # Flatten the input array and coordinates
    z = array.flatten()
    x = x.flatten()
    y = y.flatten()

    # Initial guess for the Gaussian parameters
    amplitude_guess = np.max(z)
    xo_guess, yo_guess = np.unravel_index(z.argmax(), array.shape)
    sigma_x_guess = 1.0
    sigma_y_guess = 1.0
    theta_guess = 0.0
    offset_guess = 0.0

    initial_guess = (amplitude_guess, xo_guess, yo_guess, sigma_x_guess, sigma_y_guess, theta_guess, offset_guess)

    # Fit the Gaussian using curve_fit
    popt, _ = curve_fit(_gaussian_2D, (x, y), z, p0=initial_guess)

    return popt  # popt contains the fitted parameters

def _find_connected_groups(array):
    labeled_array = measure.label(array > 0, connectivity=2)
    props = measure.regionprops(labeled_array)

    groups = []
    for prop in props:
        coords = prop.coords[:, ::-1]  # Reversing the order for (x, y) convention
        table = QTable(coords, names=('x', 'y'))
        groups.append(table)

    return groups

def _calibrate_image(image):
    """
    Calibrates an image by subtracting the median.
    """
    return image - np.median(image)

def _find_peaks(image):
    """
    Finds peaks in an image.
    """
    std = np.std(image)
    peaks = np.where(image > n_sigma*std)    
    return peaks

def _group_peaks(image,peaks):
    """
    Groups peaks together by using 2D convolution to find neighboring peaks and group them together.
    returns a list of QTables with the following columns:
        'x' - x coordinate of the peak
        'y' - y coordinate of the peak
    """
    # Create a zero image with the same shape as the input image and set the pixels corresponding to peaks to 1
    _image = np.zeros_like(image)
    _image[peaks] = 1
    
    # Define a kernel to convolve the image with
    kernel = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ])
    
    # Convolve the peak image with the kernel
    convolution = convolve(_image,kernel,boundary='extend')
    
    # Binarize the convolution, so that all pixels above 0 are 1
    convolution[convolution > 0] = 1

    # Find connected peaks to group them together into stars
    groups = _find_connected_groups(convolution)
    
    # Remove groups with less than 9 peaks (To remove single peak groups, which is probably noise)
    groups = [group for group in groups if len(group) > 8]
    
    return groups

def _get_centroids(image,groups):
    """
    Get the centroids of the groups by taking the mean of the x and y coordinates. Returns a QTable with the following columns:
        'id' - The id of the group
        'x' - x coordinate of the centroid
        'y' - y coordinate of the centroid
    """
    output = QTable(names=('id','x','y'),dtype=('i4','f4','f4'))
    for i,group in enumerate(groups):
        x = np.mean(group['x'])
        y = np.mean(group['y'])
        output.add_row((i,x,y))
        
    return output

def _fit_gaussians(image,groups,centroids):
    """
    Fits a 2D Gaussian to each group and returns the parameters of the fit in a QTable with the following columns:
        'id' - The id of the group
        'x' - x coordinate of the centroid
        'y' - y coordinate of the centroid
        'amplitude' - The amplitude of the Gaussian
        'x_mean' - The mean of the Gaussian in the x direction
        'y_mean' - The mean of the Gaussian in the y direction
        'x_stddev' - The standard deviation of the Gaussian in the x direction
        'y_stddev' - The standard deviation of the Gaussian in the y direction
        'theta' - The rotation angle of the Gaussian
    """
    output = QTable(names=('id','x','y','amplitude','x_mean','y_mean','x_stddev','y_stddev','theta','offset','fwhm'),dtype=('i4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4'))
    
    for i,group in enumerate(groups):
        # Attempt to fit a 2D Gaussian to the group. If it fails, skip the group.
        try:
            # Get a subset of the image according to window_size setting
            subset = image[int(centroids['y'][i]-window_size):int(centroids['y'][i]+window_size),int(centroids['x'][i]-window_size):int(centroids['x'][i]+window_size)]
            popt = _fit_2D_gaussian(subset)
            
            # Calculate FWHM
            fwhm = 2*np.sqrt(2*np.log(2))*np.mean([popt[3],popt[4]])
            
            output.add_row((i,centroids['x'][i],centroids['y'][i],*popt, fwhm))
        except:
            continue
    
    # Filter out stars with FWHM outside of the range
    median_fwhm = np.median(output['fwhm'])
    median_std = median_fwhm/2.355
    std_fwhm = np.std(output['fwhm'])
    output = output[(output['fwhm'] > median_fwhm - n_sigma_fwhm*std_fwhm) & (output['fwhm'] < median_fwhm + n_sigma_fwhm*std_fwhm)]
    
    # Also remove negative fwhms
    output = output[output['fwhm'] > 0]
    
    return output, median_std
    
def _aperture_photometry(image,gaussians):
    """
    Performs aperture photometry on the image using the fitted Gaussians.
    """
    aperture_radius = apperture_radius_multiplier*np.median(gaussians['fwhm'])
    # The aperture photometry function requires the coordinates to be in a list of tuples
    positions = [(gaussian['x'],gaussian['y']) for gaussian in gaussians]
    
    apertures = CircularAperture(positions,r=aperture_radius)
    fluxs = aperture_photometry(image,apertures)['aperture_sum']
    gaussians['flux'] = fluxs
    gaussians['flux_err'] = np.sqrt(fluxs)
    return gaussians

def _fix_QTable(table):
    # Drop some unneeded columns
    table = table['id','x','y','amplitude','x_stddev','y_stddev','fwhm','flux']
    return table

def extract_stars(image,return_std=False):
    """
    Extracts stars from an image.
    """
    image = _calibrate_image(image)
    peaks = _find_peaks(image)
    groups = _group_peaks(image,peaks)
    centroids = _get_centroids(image,groups)
    gaussians, std  = _fit_gaussians(image,groups,centroids)
    stars = _aperture_photometry(image,gaussians)
    stars = _fix_QTable(stars)
    if return_std:
        return stars, std
    return stars
    