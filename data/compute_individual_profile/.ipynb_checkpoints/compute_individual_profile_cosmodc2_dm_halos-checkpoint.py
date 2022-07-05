import numpy as np
import clmm
import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import edit
import glob
import time
from astropy.table import QTable, Table, vstack, join, hstack

def mask(table):
    masks = (table['mag_i'] < 24.5)*(table['mag_r'] < 28)
    return table[masks]

cosmo = clmm.Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
#name_lens_cat = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/lens_catalog_cosmoDC2_v1.1.4_redmapper_v0.8.1.pkl'
name_lens_cat = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/lens_catalog_SkySim5000.pkl'
ra_name, dec_name, z_name = 'ra', 'dec', 'redshift'
obs_name = 'M200c'
lens_cat = edit.load_pickle(name_lens_cat)
lens_cat['M200c']=lens_cat['baseDC2/sod_halo_mass']/cosmo['h']

where_bckd_catalog = '/sps/lsst/users/cpayerne/CLMassDC2/cosmoDC2/dm_halos/'
file = glob.glob(where_bckd_catalog + 'l*')
bin_edges = clmm.dataops.make_bins(0.5, 10, 15, method='evenlog10width')

names=['id', ra_name, dec_name, z_name, obs_name, 'DSt', 'DSx', 'W_l', 'radius']
ind_profile = {n:[] for n in names}
for i, name_file in enumerate(file):
    
    #cluster infos
    cluster_id = int(name_file.split('.pkl')[0].split('halo_')[1])
    lens = lens_cat[lens_cat['halo_id'] == cluster_id][0]
    ra, dec, z = lens['ra'], lens['dec'], lens['redshift']
    obs = lens[obs_name]
    
    #bckgd galaxy catalog
    table = edit.load_pickle(name_file)
    table = mask(table)
    #add masks ?
    cl = clmm.galaxycluster.GalaxyCluster('halo', ra, dec, z, clmm.gcdata.GCData(Table(table)))
    theta1, g_t, g_x = cl.compute_tangential_and_cross_components(is_deltasigma=True, cosmo=cosmo)
    #sigma_c_name = 'sigma_c'
    #sigma_c = cl.galcat[sigma_c_name]
    #cl.galcat['ds'] = sigma_c*cl.galcat['et']
    #compute weights
    #cl.galcat['w_ls'] = 1./cl.galcat[sigma_c]**2
    cl.compute_galaxy_weights(use_pdz = False,
         use_shape_noise = False, shape_component1 = 'e1', shape_component2 = 'e2', 
         use_shape_error = False, shape_component1_err = None, shape_component2_err = None, 
         weight_name = 'w_ls', cosmo = cosmo, is_deltasigma = True, add = True)

    cl.galcat['radius'] = cosmo.eval_da_z1z2(0, z)*cl.galcat['theta']
    
    
    ce = clmm.ClusterEnsemble('id', [])
    
    p = ce.make_individual_radial_profile(cl, 'Mpc', bins=bin_edges, error_model='ste',
                                       cosmo=cosmo, tan_component_in='et', cross_component_in='ex',
                                       tan_component_out='gt', cross_component_out='gx',
                                       tan_component_in_err=None, cross_component_in_err=None,
                                       weights_in='w_ls', weights_out='W_l')
    data = ce.data[0]

    data_to_save = [cluster_id, ra, dec, z, obs, data['gt'], data['gx'], data['W_l'], data['radius']]
    for s, n in enumerate(names): ind_profile[n].append(data_to_save[s])
    #if i > 1000: break

edit.save_pickle(Table(ind_profile), '/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/data_new_version/ind_profile_dm_halos.pkl')