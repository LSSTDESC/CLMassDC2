import numpy as np
import pickle,sys
from astropy.table import Table, hstack, vstack
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/data_extraction')
import extract_in_dc2 as dc2
import photoz_utils as utils
import convert_shapes as shape
import GCRCatalogs
from GCR import GCRQuery
import clmm
import glob
from clmm import Cosmology
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import edit

start, end = int(sys.argv[1]), int(sys.argv[2])

#select galaxy clusters
lens_catalog_name='/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/lens_catalog_cosmoDC2_v1.1.4_redmapper_v0.8.1.pkl'
#lens_catalog_name='/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/lens_catalog_SkySim5000.pkl'

lens_catalog=edit.load_pickle(lens_catalog_name)

#select subsample of clusters SkySim5000
# mask = (lens_catalog['baseDC2/sod_halo_mass']/cosmo['h'] > 1e14)*(lens_catalog['redshift'] < 1.2)*(lens_catalog['redshift'] > .2)
# lens_catalog = lens_catalog[mask]
# lens_catalog['cluster_id'] = lens_catalog['halo_id']

where_to_save='/sps/lsst/users/cpayerne/CLMassDC2/DC2/'

#select subsample of clusters redMaPPer#
mask_select = (lens_catalog['richness'] > 20)*(lens_catalog['redshift'] > .2)
lens_catalog = lens_catalog[mask_select]
mask_n=np.arange(start, end)
lens_catalog_truncated=lens_catalog[mask_n]

file_already_saved = glob.glob(where_to_save + 'l*')
cluster_id_saved = []
for f in file_already_saved:
    cluster_id_saved.append(int(f.split('.pkl')[0].split('halo_')[1]))
mask_saved = np.isin(lens_catalog_truncated['cluster_id'], cluster_id_saved)
lens_catalog_truncated = lens_catalog_truncated[np.invert(mask_saved)] 
print(len(lens_catalog_truncated))
print('--')
#select galaxy catalogs
cat = GCRCatalogs.load_catalog('dc2_object_run2.2i_dr6_v2_with_addons')
z_bins= cat.photoz_pdf_bin_centers
z_bins[0]=1e-7
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
                           'ext_shapeHSM_HsmShapeRegauss_resolution',
                           'mcal_g1', 'mcal_g2', 'mcal_gauss_g1','mcal_gauss_g1', 
                           'mcal_psf_g1', 'mcal_psf_g2']
    quantity_wanted_photozs = ['photoz_mean', 'photoz_pdf','photoz_odds']
    return  [quantity_label, quantity_wanted_HSM_Metacal, quantity_wanted_photozs]

ql, qs, qz = quantities()[0], quantities()[1], quantities()[2]
to_save = ql + qs
pz_to_save = ['sigma_c_photoz', 'p_background', 'photoz_err', 
              'sigma_c_photoz_estimate_0', 'sigma_c_photoz_estimate_1', 
              'sigma_c_photoz_estimate_2', 
              'z_estimate_0', 'z_estimate_1', 'z_estimate_2', 
              'galaxy_id', 
              'photoz_mean', 'photoz_mode', 'photoz_odds']


for n, lens in enumerate(lens_catalog_truncated):
    #select single cluster
    z, ra, dec = lens['redshift'], lens['ra'], lens['dec']
    print(z, ra, dec)
    cluster_id = lens['cluster_id']
    richness = lens['richness']
    name_cat = 'lensing_catalog_halo_' + str(cluster_id)
    name_full_cat = where_to_save + name_cat + '.pkl'
    if name_full_cat in glob.glob(where_to_save + '*'):
        print('already saved')
        continue
    #maximum raduis to extract
    rmax = 10.1
    i = 0
    #extract with GCR
    distance_to_cluster = cosmo.eval_da(z)
    theta_max = (rmax/distance_to_cluster) * (180./np.pi)
    coord_filters = [f'({ra} - ra)**2 *cos({dec}*{np.pi}/180)**2 + (dec - {dec})**2 <= {theta_max}**2']
    tracts=dc2.neighboring_tract(ra, dec, theta_max)
    print(tracts)
    for i, tract in enumerate(tracts):
        #browse neighboring tracts
        q = ql + qz + qs
        object_data = Table(cat.get_quantities(q, filters=(coord_filters + filters(zmin=z)),
                                                native_filters = ['tract == ' + str(tract)]))
        if len(object_data['ra'])==0: continue
        
        #rename HSM-chi shape definition
        object_data['chi1_HSM'], object_data['chi2_HSM'] = object_data['ext_shapeHSM_HsmShapeRegauss_e1'], object_data['ext_shapeHSM_HsmShapeRegauss_e2']
        object_data['chi_HSM'] = np.sqrt(object_data['chi1_HSM']**2 + object_data['chi2_HSM']**2)
        object_data['chi_sigma_HSM'] = object_data['ext_shapeHSM_HsmShapeRegauss_sigma']
        object_data['chi_HSM_resolution'] = object_data['ext_shapeHSM_HsmShapeRegauss_resolution']
        #compute epsilon shapes
        object_data['e1_HSM'], object_data['e2_HSM'] = shape.e12(object_data['chi1_HSM'], object_data['chi2_HSM'])
        object_data['e_HSM'] = np.sqrt(object_data['e1_HSM']**2 +object_data['e2_HSM']**2) 
        object_data['e_sigma_HSM'] = shape.e_sigma(object_data['chi1_HSM'], object_data['chi2_HSM'], object_data['id'])
        #compute photozs quantities
        print('pz')
        pzbins_table=np.array([z_bins for i in range(len(object_data['photoz_pdf'].data))])
        pz_quantities = utils.compute_photoz_quantities(z, object_data['photoz_pdf'], 
                                                                pzbins_table, n_samples_per_pdf=3, 
                                                               cosmo=cosmo,use_clmm=False)
        object_data.remove_column('photoz_pdf')
        object_data = hstack([object_data, pz_quantities])  
        if i==0:
            names = object_data.colnames
            table = Table(names = names)
            table = vstack([table, object_data])
            i = 1
        else: table = vstack([table, object_data])
        
    try: a = table
    except: continue
     
    #edit.save_pickle(table, where_to_save + name_cat)
