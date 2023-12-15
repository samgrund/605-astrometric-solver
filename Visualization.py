import numpy as np
import matplotlib.pyplot as plt
from Utilities import graham_scan, generate_high_visibility_colors

# Get astropy plotting style
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style) # Use astropy plotting style

def compare_hashes(local_hash,gaia_hash):
    """
    Visually compares two hashes.
    """
    coords_1 = [(0,0),(local_hash['xc'],local_hash['yc']),(local_hash['xd'],local_hash['yd']),(local_hash['xb'],local_hash['yb'])]
    coords_2 = [(0,0),(gaia_hash['xc'],gaia_hash['yc']),(gaia_hash['xd'],gaia_hash['yd']), (gaia_hash['xb'],gaia_hash['yb'])]
    
    max_x = max([local_hash['xc'],local_hash['xd'],local_hash['xb'],gaia_hash['xc'],gaia_hash['xd'],gaia_hash['xb']])
    max_y = max([local_hash['yc'],local_hash['yd'],local_hash['yb'],gaia_hash['yc'],gaia_hash['yd'],gaia_hash['yb']])
    max_coord = max(max_x,max_y)
    
    gaia_idx = gaia_hash['index']
    local_idx = local_hash['index']
    
    coords_1 = graham_scan(coords_1)
    coords_2 = graham_scan(coords_2)
        
    fig,ax = plt.subplots(figsize=(5,4),tight_layout=True)
    ax.set_xlabel('x')
    ax.set_ylim(0 - max_coord*0.1,max_coord*1.1)
    ax.set_xlim(0-max_coord*0.1,max_coord*1.1)
    ax.set_ylabel('y')
    ax.scatter([0,local_hash['xc'],local_hash['xd'],local_hash['xb']],[0,local_hash['yc'],local_hash['yd'],local_hash['yb']],color='r',label='LOCAL')
    ax.scatter([0,gaia_hash['xc'],gaia_hash['xd'],gaia_hash['xb']],[0,gaia_hash['yc'],gaia_hash['yd'],gaia_hash['yb']],color='b',label='GAIA')
    ax.legend()
    
    
    ax.add_patch(plt.Polygon(coords_1,fill=False,color='r'))
    ax.add_patch(plt.Polygon(coords_2,fill=False,color='b'))
    
    return fig,ax

def plot_stellar(image,skycoords=False,stretch=True,vlims=[1,99],dpi=300,cmap='gray_r'):
    """
    Plots the image of the field with a nice scaling.
    """
    fig,ax = plt.subplots(dpi=dpi)
    if stretch:
        image = np.arcsinh(image)
        vlims = np.percentile(image,vlims)
    else:
        vlims = np.percentile(image,vlims)
    ax.imshow(image,origin='lower',cmap=cmap,interpolation='nearest',vmin=vlims[0],vmax=vlims[1])
    ax.grid(visible=False)
    if skycoords:
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
    else:
        ax.axis('off')
    return fig,ax
    
def visualize_with_wcs(image,wcs,vlims=[1,99]):
    fig,ax = plt.subplots(subplot_kw={'projection':wcs})

    vlims = np.percentile(image,vlims)

    ax.imshow(image,vmin=vlims[0],vmax=vlims[1],origin='lower',cmap='gray_r')

    ax.coords.grid(True, color='white', ls='solid')
    ax.coords[0].set_axislabel('Galactic Longitude')
    ax.coords[1].set_axislabel('Galactic Latitude')

    overlay = ax.get_coords_overlay('fk5')
    overlay.grid(color='white', ls='dotted')
    overlay[0].set_axislabel('Right Ascension (J2000)')
    overlay[1].set_axislabel('Declination (J2000)')
    ax.set_xlim(0,image.shape[1])
    ax.set_ylim(0,image.shape[0])
    
    return fig,ax
    