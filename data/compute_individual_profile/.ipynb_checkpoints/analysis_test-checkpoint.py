import numpy as np
import pickle, sys
import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
from clmm.support import mock_data as mock

def load(filename, **kwargs):
    """Loads GalaxyCluster object to filename using Pickle"""
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)

where_catalogs = '/sps/lsst/users/cpayerne/RedMapper_clusters/cosmoDC2_v1.1.4_image_20_Mpc/all_11_Mpc/'
cluster_infos = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/Galaxy_Cluster_Catalogs_details/cosmoDC2/RedMapper_galaxy_clusters.pkl')
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]


Mass_bin = [[10**13, 10**15]]
Richness_bin = [[10, 3000]]
Z_bin = [[0.2, 1]]

r"""
check catalogs
"""

redshift_def = 'redshift'
Obs_bin = Richness_bin
obs_def = 'richness'
cluster_name = 'cluster_id'
cluster_key = 'cluster_id_RedMapper_'
column_to_extract = ['dc2_galaxy_id', 'et', 'ex', 'halo_id', 'sigma_c_1_TRUE', '<sigma_c-1>_flexzboost_v1', '<sigma_c-1>_v1']

r"""
make_Galaxy_Cluster_Catalog
"""

def quantity_modifier(cl): 
    mask = cl.galcat['z_odds_photoz_v1'] > 0.8
    data_cut = clmm.GCData(cl.galcat[mask])
    cl_cut = clmm.GalaxyCluster('Stack', cl.ra, cl.dec, cl.z, data_cut)
    return cl_cut

z_gal_name = 'z_mean_photoz_flexzboost_v1'
cluster_ra = 'ra'
cluster_dec = 'dec'
shape_component1_in = 'e1_cosmodc2'
shape_component2_in = 'e2_cosmodc2'                                           
tan_component_out = 'et'
cross_component_out = 'ex'
rmax = 10
modify_quantity = False
sigma_c_1_in ='<sigma_c-1>_flexzboost_v1'
err_shape_in = None

r"""Estimating shear profile"""

down, up, n_bins = 0.1 , rmax, 30
bin_edges = make_bins(down, up, nbins=n_bins, method='evenlog10width')
radial_bin = binning(bin_edges)
col_to_bin = ['w_ls','halo_id','st', 'sx', 'r']

col_to_average = ['st', 'sx', 'r']
name_averaged_cl = ['gt_av', 'gx_av', 'radius_av']

col_to_sum = ['w_ls']
name_sum_cl = ['norm_sum']
extract_extra = []
r"""
Saving file
"""
compute_selection_response = False
save = True

where_to_save = '/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/p_RedMapper_clusters/Run_Python_Codes/compute_individual_ds/'

filename = 'test.pkl'

name = where_to_save + filename



