import numpy as np
import pickle,sys
from astropy.table import Table, hstack
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/data_extraction')
import extract_in_dc2 as dc2
import photoz_utils as utils
import matplotlib.pyplot as plt
import GCRCatalogs
from GCR import GCRQuery
import healpy
import clmm
from clmm.dataops import compute_galaxy_weights
from clmm import Cosmology
from scipy.integrate import simps
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import edit

start, end = int(sys.argv[1]), int(sys.argv[2])

#select galaxy clusters
lens_catalog_name='/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/lens_catalog_cosmoDC2_v1.1.4_redmapper_v0.8.1.pkl'
lens_catalog=edit.load_pickle(lens_catalog_name)
where_to_save='/sps/lsst/users/cpayerne/CLMassDC2/DC2/'
mask=np.arange(start, end)
lens_catalog_truncated=lens_catalog[mask]

#select galaxy catalogs
cat = GCRCatalogs.load_catalog('dc2_object_run2.2i_dr6_v2_with_addons')
#filters
def filters(zmin=.1):
    #redshift + quality filters
    
    resolution_min = 0.2
    err_shape_max = 0.4
    snr_i_cModel_min = 10
    
    object_basic_cuts = [
        # Extended objects
        GCRQuery('extendedness > 0'),     
        GCRQuery('clean'), 
        # The source has no flagged pixels (interpolated, saturated, edge, clipped...) 
        GCRQuery('xy_flag == 0'),                                     
        # Flag for bad centroid measurement
        GCRQuery('ext_shapeHSM_HsmShapeRegauss_flag == 0'),            
        # Error code returned by shape measurement code
        GCRQuery((np.isfinite, 'ext_shapeHSM_HsmShapeRegauss_sigma')),] 
        # Shape measurement uncertainty should not be NaN
    object_properties_cuts = [
        GCRQuery(f'snr_i_cModel > {snr_i_cModel_min}'),                              
        # SNR > 10
        GCRQuery('mag_i_cModel < 25'),                                
        # cModel imag brighter than 24.5
        GCRQuery(f'ext_shapeHSM_HsmShapeRegauss_resolution >= {resolution_min}'), 
        # Sufficiently resolved galaxies compared to PSF
        GCRQuery(f'ext_shapeHSM_HsmShapeRegauss_sigma <= {err_shape_max}'), 
        # New cut on blendedness:
        GCRQuery('blendedness < 10**(-0.375)'),                      
        # Avoid spurious detections and those contaminated by blends
        GCRQuery(f'photoz_mean > {zmin}'),
        #GCRQuery(f'photoz_mean < {zmax}')
        ]
    return  object_basic_cuts + object_properties_cuts

#quantities
def quantities():
    #quantities to extract
    quantity_label = ['id', 'ra', 'dec', 'mag_i_cModel','snr_i_cModel', 'mag_i']
    quantity_wanted_HSM_Metacal = ['ext_shapeHSM_HsmShapeRegauss_e2',
                           'ext_shapeHSM_HsmShapeRegauss_e1',
                           'ext_shapeHSM_HsmShapeRegauss_sigma', 
                           'ext_shapeHSM_HsmShapeRegauss_resolution',]
                            #'mcal_g1', 'mcal_g2', 'mcal_gauss_g1','mcal_gauss_g1', 'mcal_psf_g1', 'mcal_psf_g2']
    quantity_wanted_photozs = ['photoz_mean', 'photoz_pdf','photoz_odds']
    return  quantity_label + quantity_wanted_HSM_Metacal + quantity_wanted_photozs

#extraction
for n, lens in enumerate(lens_catalog_truncated):
    #select single cluster
    z, ra, dec = lens['redshift'], lens['ra'], lens['dec']
    cluster_id = lens['cluster_id']
    print(cluster_id)
    richness = lens['richness']
    
    #maximum raduis to extract
    rmax = 1
    
    #extract with GCR
    dat_extract=dc2.extract(lens_redshift=z, lens_ra=ra, lens_dec=dec, rmax=rmax, cosmo=cosmo, 
               GCRcatalog=cat, data_filters=filters(zmin=z + .05), quantities=quantities())
    
    #compute photoz WL quantities
    pz_quantities = utils.compute_photoz_quantities(z, dat_extract['photoz_pdf'], dat_extract['pzbins'], 
                                                           n_samples_per_pdf=3, cosmo=cosmo, use_clmm=False)
    
    dat_full = hstack([Table(dat_extract), Table(pz_quantities)])
    dat_full.remove_column('pzbins')
    dat_full.remove_column('photoz_pdf')
    cl_full = clmm.GalaxyCluster('dc2_weak_lensing_catalog', ra, dec, z, clmm.GCData(dat_full))
    
    name = where_to_save + 'redmapper_cluster_id_' + str(cluster_id) + '.pkl'
    
    edit.save_pickle(cl_full, name)
