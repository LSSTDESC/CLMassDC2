import sys
import os
import numpy as np
from astropy.table import QTable, Table, vstack, hstack
import pickle 
import glob
import time,math
import clmm
import GCRCatalogs
from GCR import GCRQuery
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
import numpy as np
import DC2_tract_coordinates as c
import lens_data as run

def combinatory(X):
    dimension = len(X)
    mesh = np.array(np.meshgrid(*X))
    combinations = mesh.T.reshape(-1, dimension)
    Xcomb = combinations.T

cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
object_cat = GCRCatalogs.load_catalog('dc2_object_run2.2i_dr6_v2_with_addons')
z_bins = object_cat.photoz_pdf_bin_centers
ra_u, dec_u = (80-40)*np.random.random(8000) + 40, (-20 + 50)*np.random.random(5000) - 50
ra_comb, dec_comb = combinatory([ra_u, dec_u])

def e12(chi1, chi2):
    
    chi = np.sqrt(chi1**2 + chi2**2)
    zero = 1. + (1. - chi**2)**(1./2.)
    return chi1/zero, chi2/zero
    
def chi12(e1, e2):
    
    e = np.sqrt(e1**2+e2**2)
    zero = 1. + e**2
    return 2*e1/zero, 2*e2/zero
    
def e_sigma(chi1, chi2, chi_sigma):
    
    chi = np.sqrt(chi1**2 + chi2**2)
    zero = 1. + (1. - chi**2)**(1/2)
    first = 1./zero
    second = chi**2/((1 - chi**2)**0.5)
    return first*(1 + second*first)*chi_sigma

def chi_sigma(e1, e2, e_sigma_):
    
    e = np.sqrt(e1**2+e2**2)
    chi1, chi2 = chi12(e1, e2)
    chi = np.sqrt(chi1**2 +chi2**2)
    return e_sigma_*chi*(1./e - chi)

def quantity_wanted():
    r"""query"""
    quantity_label = ['id', 'ra', 'dec', 'mag_i_cModel','snr_i_cModel', 'mag_i']
    quantity_wanted_HSM_Metacal = ['ext_shapeHSM_HsmShapeRegauss_e2',
                           'ext_shapeHSM_HsmShapeRegauss_e1',
                           'ext_shapeHSM_HsmShapeRegauss_sigma', 
                           'ext_shapeHSM_HsmShapeRegauss_resolution',
                          'mcal_g1', 'mcal_g2', 'mcal_gauss_g1','mcal_gauss_g1', 'mcal_psf_g1', 'mcal_psf_g2']
    quantity_wanted_photozs = ['photoz_mean', 'photoz_pdf']
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
        GCRQuery('photoz_mean > 0.3'),
        GCRQuery('photoz_mean < 1')]
    return  object_basic_cuts + object_properties_cuts

def angular_distance(ra_cl, dec_cl, ra, dec):
    return np.sqrt((ra_cl - ra)**2*np.cos(dec_cl*np.pi/180)**2 + (dec - dec_cl)**2)
    
def extract(n):
    z, ra, dec = lens_catalog[n][run.redshift_name], lens_catalog[n][run.dec_name], lens_catalog[n][run.dec_name]
    da = cosmo.eval_da(z)
    theta_max = (15./da) * (180./np.pi)
    dist = angular_distance(ra_cl, dec_cl, ra_comb, dec_comb)
    mask = [dist <= theta_max]
    ra_comb_cut, dec_comb_cut = ra_comb[mask], dec_comb[mask]
    neighboring_tracts = c.neigboring_tracts(ra_deg = ra_comb_cut, dec_deg = dec_comb_cut)
    coord_filters = [f'({ra_cl} - ra)**2*cos({dec_cl}*{np.pi}/180)**2 + (dec - {dec_cl})**2 <= {theta_max}**2']
    quantity = quantity_wanted()
    filters = filters(z, 3.)
    for i,tr in enumerate(neighboring_tracts['tract_id']):
        object_data = object_cat.get_quantities(quantity, filters=(coord_filters + filters),
                                                native_filters = ['tract == ' + str(tr)])
        t = Table(object_data)
        if i == 0:
            dat = t
            continue
            dat = vstack([t,dat])
        if len(v) == 0:
            return 0

    dat['chi1_HSM'], dat['chi2_HSM'] = dat['ext_shapeHSM_HsmShapeRegauss_e1'], dat['ext_shapeHSM_HsmShapeRegauss_e2']
    dat['chi_HSM'] = np.sqrt(dat['chi1_HSM']**2 + dat['chi2_HSM']**2)
    dat['chi_sigma_HSM'] = dat['ext_shapeHSM_HsmShapeRegauss_sigma']
    
    dat['e1_HSM'], dat['e2_HSM'] = e12(dat['chi1_HSM'], v['chi2_HSM'])
    dat['e_HSM'] = np.sqrt(dat['e1_HSM']**2 + dat['e2_HSM']**2) 
    dat['e_sigma_HSM'] = e_sigma(dat['chi1_HSM'], dat['chi2_HSM'], dat['id'])
    dat['chi_HSM_resolution'] = dat['ext_shapeHSM_HsmShapeRegauss_resolution']
    dat['pzbins'] = np.array([z_bins for i, ide in enumerate(dat['galaxy_id'])])
    cl_full = GalaxyCluster('photoz', ra, dec, z, dat)
    #compute weights
    cl_full.compute_galaxy_weights(z, cosmo, z_source=None, pzpdf='photoz_pdf', 
                                   pzbins='pzbins',
                                   shape_component1=None, shape_component2=None,
                                   shape_component1_err=None, shape_component2_err=None,
                                   p_background=None, use_shape_noise=False, is_deltasigma=True,
                                   validate_input=True)
    cl_full_remove = cl_full.galcat
    #remove useless columns
    cl_full_remove.remove_column('pzbins')
    cl_full_remove.remove_column('photoz_pdf')
    cl_full = GalaxyCluster(photoz, ra, dec, z, cl_full_remove)
    return cl_full