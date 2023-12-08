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
    coords_1 = [(0,0),(local_hash['xc'],local_hash['yc']),(local_hash['xd'],local_hash['yd']),(1,1)]
    coords_2 = [(0,0),(gaia_hash['xc'],gaia_hash['yc']),(gaia_hash['xd'],gaia_hash['yd']),(1,1)]
    
    gaia_idx = gaia_hash['index']
    local_idx = local_hash['index']
    
    coords_1 = graham_scan(coords_1)
    coords_2 = graham_scan(coords_2)
        
    fig,ax = plt.subplots(figsize=(5,5),tight_layout=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter([0,local_hash['xc'],local_hash['xd'],1],[0,local_hash['yc'],local_hash['yd'],1],color='r',label=f'Local hash ({local_idx})')
    ax.scatter([0,gaia_hash['xc'],gaia_hash['xd'],1],[0,gaia_hash['yc'],gaia_hash['yd'],1],color='b',label=f'GAIA hash ({gaia_idx})')
    
    ax.legend()

    ax.add_patch(plt.Polygon(coords_1,closed=True,fill=False,linewidth=1.5, alpha=.8, color='r'))
    ax.add_patch(plt.Polygon(coords_2,closed=True,fill=False,linewidth=1.5, alpha=.8,color='b'))
    plt.show()

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
    
    