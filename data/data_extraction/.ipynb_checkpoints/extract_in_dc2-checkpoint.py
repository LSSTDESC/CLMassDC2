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
import extract_in_cosmodc2 as extract_utils
import convert_shapes as shape

def combinatory(X):
    dimension = len(X)
    mesh = np.array(np.meshgrid(*X))
    combinations = mesh.T.reshape(-1, dimension)
    Xcomb = combinations.T
    return Xcomb

def query():
    r"""query"""
    quantity_label = ['id', 'ra', 'dec', 'mag_i_cModel','snr_i_cModel', 'mag_i']
    quantity_wanted_HSM_Metacal = ['ext_shapeHSM_HsmShapeRegauss_e2',
                           'ext_shapeHSM_HsmShapeRegauss_e1',
                           'ext_shapeHSM_HsmShapeRegauss_sigma', 
                           'ext_shapeHSM_HsmShapeRegauss_resolution',
                          'mcal_g1', 'mcal_g2', 'mcal_gauss_g1','mcal_gauss_g1', 'mcal_psf_g1', 'mcal_psf_g2']
    quantity_wanted_photozs = ['photoz_mean', 'photoz_pdf','photoz_odds']
    return  quantity_label + quantity_wanted_HSM_Metacal + quantity_wanted_photozs

def filters(zmin, zmax):
    r"""query"""
    object_basic_cuts = [
        GCRQuery('extendedness > 0'),     # Extended objects
        GCRQuery('clean'), # The source has no flagged pixels (interpolated, saturated, edge, clipped...) 
        GCRQuery('xy_flag == 0'),                                      # Flag for bad centroid measurement
        GCRQuery('ext_shapeHSM_HsmShapeRegauss_flag == 0'),            # Error code returned by shape measurement code
        GCRQuery((np.isfinite, 'ext_shapeHSM_HsmShapeRegauss_sigma')),] # Shape measurement uncertainty should not be NaN
    object_properties_cuts = [
        GCRQuery('snr_i_cModel > 10'),                              # SNR > 10
        GCRQuery('mag_i_cModel < 25'),                                 # cModel imag brighter than 24.5
        GCRQuery('ext_shapeHSM_HsmShapeRegauss_resolution >= 0.4'), # Sufficiently resolved galaxies compared to PSF
        GCRQuery('ext_shapeHSM_HsmShapeRegauss_sigma <= 0.4'),      # Shape measurement errors reasonable
        # New cut on blendedness:
        GCRQuery('blendedness < 10**(-0.375)'),                      # Avoid spurious detections and those contaminated by blends
        GCRQuery(f'photoz_mean > {zmin}'),
        GCRQuery(f'photoz_mean < {zmax}')]
    return  object_basic_cuts + object_properties_cuts

def angular_distance(ra_cl, dec_cl, ra, dec):
    r"""anbgular dustance between lens and gal"""
    return np.sqrt((ra_cl - ra)**2*np.cos(dec_cl*np.pi/180)**2 + (dec - dec_cl)**2)

def neighboring_tract(lens_ra, lens_dec, theta_max):

    ra_ = lens_ra + np.random.random(100)*(theta_max + theta_max) - theta_max
    dec_ = lens_dec + np.random.random(100)*(theta_max + theta_max) - theta_max
    plt.scatter(ra_, dec_)
    return c.neigboring_tracts(ra_deg = ra_, dec_deg = dec_)
    
def extract(lens_redshift=None, lens_ra=None, lens_dec=None, rmax=None, cosmo=None, GCRcatalog=None):
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
    coord_filters = [f'({lens_ra} - ra)**2*cos({lens_dec}*{np.pi}/180)**2 + (dec - {lens_dec})**2 <= {theta_max}**2']
    quantity = query()
    filters_used = filters(lens_redshift, 3.)
    tracts=neighboring_tract(lens_ra, lens_dec, theta_max)
    for i,tract in enumerate(tracts):
        object_data = GCRcatalog.get_quantities(quantity, filters=(coord_filters + filters_used),
                                                native_filters = ['tract == ' + str(tract)])
        t = Table(object_data)
        if i == 0:
            dat = t
            continue
        else: dat = vstack([t,dat])
    #rename chi
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
    #compute weights
    w_ls_photoz = extract_utils.compute_photoz_weights(lens_redshift, 
                                                       dat['photoz_pdf'],
                                                       dat['pzbins'], 
                                                       cosmo=cosmo, 
                                                       use_clmm=False)
    dat['w_ls_BPZ']=w_ls_photoz
    cl_full = GalaxyCluster('dc2', lens_ra, lens_dec, lens_redshift, clmm.GCData(dat))
    cl_full_remove = cl_full.galcat
    cl_full_remove.remove_column('pzbins')
    cl_full_remove.remove_column('photoz_pdf')
    return cl_full