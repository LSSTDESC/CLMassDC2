import sys, os, glob, pickle
import numpy as np
from astropy.table import Table
import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
sys.path.append('/pbs/throng/lsst/users/cpayerne/GitForThesis/DC2Analysis/modeling/')
import CL_WL_DATAOPS_make_MetaCatalog_test as metacat
import CL_WL_DATAOPS_make_binned_profile as prf

path = './'

import importlib.util
spec = importlib.util.spec_from_file_location("analysis", path + str(sys.argv[1]))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

def load(filename, **kwargs):
    """Loads GalaxyCluster object to filename using Pickle"""
    with open(filename, 'rb') as fin:   
        return pickle.load(fin, **kwargs)

norme, z, obs, cluster_name, cluster_id = [], [], [], [], []
radial_bins, radius, gt, gx = [], [], [], []

print(f'{analysis.Z_bin[0][0]:.2f} < z < {analysis.Z_bin[0][1]:.2f}')
print(f'{analysis.Obs_bin[0][0]:.2f} < obs < {analysis.Obs_bin[0][1]:.2f}')

cat = metacat.MetaCatalog(cosmo = analysis.cosmo)
cat._check_available_catalogs(dc2_infos = analysis.cluster_infos, 
                              redshift_def = analysis.redshift_def, 
                              obs_def = analysis.obs_def,
                              z_bin = analysis.Z_bin[0], 
                              obs_bin = analysis.Obs_bin[0], 
                              cluster_name = analysis.cluster_name,
                              cluster_key = analysis.cluster_key,
                              where_source = analysis.where_catalogs)

print('======> n = ' + str(len(cat.file_in_bin)))
print(f'----- mean z = {cat.z:.2f}')
print(f'----- mean obs = {cat.obs:.2e}')

infos_cluster = ['cluster_id', analysis.obs_def, 'cluster_z', 'radial_bin']
col_analysis = analysis.name_averaged_cl + analysis.name_sum_cl + analysis.extract_extra
col_to_extract = infos_cluster + col_analysis
extract = {name : [] for name in col_to_extract}

for i, halo in enumerate(cat.selected_halo_id):

    p = cat.dc2_infos[cat.dc2_infos[analysis.cluster_name] == halo]

    cat_individual = cat
    cat_individual.selected_halo_id = [halo]
    cat_individual.make_GalaxyCluster_catalog(z_gal_name = analysis.z_gal_name, 
                                              cluster_ra = analysis.cluster_ra, cluster_dec = analysis.cluster_dec,
                               shape_component1_in = analysis.shape_component1_in, shape_component2_in = analysis.shape_component2_in, 
                               tan_component_out = analysis.tan_component_out, cross_component_out = analysis.cross_component_out, 
                               column_to_extract = analysis.column_to_extract,
                               rmax = analysis.rmax,
                               modify_quantity = analysis.modify_quantity, 
                               quantity_modifier = analysis.quantity_modifier)

    cl = cat_individual.cl
    cl = cat_individual.compute_signal(sigma_c_1_in = analysis.sigma_c_1_in, 
                            tan_in = analysis.tan_component_out, cross_in = analysis.cross_component_out, 
                            tan_out = 'st', cross_out = 'sx', catalog = cl)
    cl = cat_individual.compute_weights(err_shape_in = analysis.err_shape_in,  sigma_c_1_in = analysis.sigma_c_1_in, weight_out = 'w_ls',  catalog = cl)
    mask = ( cl.galcat['w_ls'] != 0 ) * np.invert( np.isnan(cl.galcat['st']) )
    cl_cut = cl.galcat[mask]
    cl_cut = clmm.GalaxyCluster('Stack', cl.ra, cl.dec, cat_individual.z, cl_cut)
    cl_cut.n_stacked_catalogs = 1
    n_gal = len(cl_cut.galcat)
    r"""Estimating shear profile"""
    shear = prf.Shear(is_deltasigma = True, cosmo = analysis.cosmo)
    profile = shear.make_binned_profile(metacatalog = cl_cut, 
                                        tan_in = 'st', cross_in = 'sx', 
                                        weights = 'w_ls', 
                                        tan_out = 'gt', cross_out = 'gx',
                                        add_columns_to_bin = analysis.col_to_bin, 
                                        bin_edges = analysis.bin_edges)
    for k, name in enumerate(analysis.col_to_average):
        shear.make_binned_average(profile = profile, v_in = name, weights = 'w_ls', v_out = analysis.name_averaged_cl[k])
    for m, name in enumerate(analysis.col_to_sum):
        shear.make_binned_sum(profile = profile, v_in = name, v_out = analysis.name_sum_cl[m])
    if analysis.compute_selection_response == True:
        shear.calculate_binned_selection_response(profile = profile,  
                                            uncal_e1 = analysis.shape_component1_in, 
                                            uncal_e2 = analysis.shape_component2_in,
                                            R_s_out = analysis.selection_response_out, 
                                            s2n_cut = analysis.s2n_cut)
    data_cluster_infos = [halo, p[analysis.obs_def], p[analysis.redshift_def], analysis.bin_edges]
    data_analysis = [profile[name] for name in col_analysis]
    data_to_extract = data_cluster_infos + data_analysis 
    for j, name in enumerate(list(extract.keys())):
        extract[name].append(data_to_extract[j])
    if analysis.save == True:
        with open(analysis.name, 'wb') as file:
            pickle.dump(Table(extract), file)
    else: continue
