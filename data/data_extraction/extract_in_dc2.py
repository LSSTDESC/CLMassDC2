import numpy as np
from astropy.table import QTable, Table, vstack, hstack
import clmm
import matplotlib.pyplot as plt
from GCR import GCRQuery
from clmm.galaxycluster import GalaxyCluster
import numpy as np
import random
import DC2_tract_coordinates as c
import lens_data as run
import convert_shapes as shape
import photoz_utils

def combinatory(X):
    dimension = len(X)
    mesh = np.array(np.meshgrid(*X))
    combinations = mesh.T.reshape(-1, dimension)
    Xcomb = combinations.T
    return Xcomb

def angular_distance(ra_cl, dec_cl, ra, dec):
    r"""anbgular dustance between lens and gal"""
    return np.sqrt((ra_cl - ra)**2*np.cos(dec_cl*np.pi/180)**2 + (dec - dec_cl)**2)

def neighboring_tract(lens_ra, lens_dec, theta_max):

    ra = lens_ra + np.random.random(100)*(theta_max + theta_max) - theta_max
    dec = lens_dec + np.random.random(100)*(theta_max + theta_max) - theta_max
    return c.neigboring_tracts(ra_deg = ra, dec_deg = dec)
    
def extract(lens_redshift=None, lens_ra=None, lens_dec=None, 
            rmax=None, cosmo=None, GCRcatalog=None, 
            data_filters=None, quantities=None):
    r"""
    extract background galaxy catalog
    Attributes:
    -----------
    lens_redshift: float
        lens redshift
    lens_ra: float
        lens right ascension
    lens_dec: float
        lens declinaison
    Returns:
    --------
    cl: GalaxyCluster
        background galaxy catalog
    """
    z_bins= GCRcatalog.photoz_pdf_bin_centers
    z_bins[0]=1e-7
    da = cosmo.eval_da(lens_redshift)
    theta_max = (rmax/da) * (180./np.pi)
    coord_filters = [f'({lens_ra} - ra)**2 *cos({lens_dec}*{np.pi}/180)**2 + (dec - {lens_dec})**2 <= {theta_max}**2']
    tracts=neighboring_tract(lens_ra, lens_dec, theta_max)
    
    for i,tract in enumerate(tracts):
        #browse neighboring tracts
        object_data = GCRcatalog.get_quantities(quantities, filters=(coord_filters + data_filters),
                                                native_filters = ['tract == ' + str(tract)])
        t = Table(object_data)
        if i == 0:
            dat = t
            continue
        else: dat = vstack([t,dat])
        
    #rename HSM-chi shape definition
    dat['chi1_HSM'], dat['chi2_HSM'] = dat['ext_shapeHSM_HsmShapeRegauss_e1'], dat['ext_shapeHSM_HsmShapeRegauss_e2']
    dat['chi_HSM'] = np.sqrt(dat['chi1_HSM']**2 + dat['chi2_HSM']**2)
    dat['chi_sigma_HSM'] = dat['ext_shapeHSM_HsmShapeRegauss_sigma']
    dat['chi_HSM_resolution'] = dat['ext_shapeHSM_HsmShapeRegauss_resolution']
    #compute epsilon shapes
    dat['e1_HSM'], dat['e2_HSM'] = shape.e12(dat['chi1_HSM'], dat['chi2_HSM'])
    dat['e_HSM'] = np.sqrt(dat['e1_HSM']**2 + dat['e2_HSM']**2) 
    dat['e_sigma_HSM'] = shape.e_sigma(dat['chi1_HSM'], dat['chi2_HSM'], dat['id'])
    #photoz
    dat['pzbins'] = np.array([z_bins for i, ide in enumerate(dat['id'])])
    return dat