import sys, os, glob, time

import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
from clmm.support import mock_data as mock

cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)

import numpy as np
from astropy.table import QTable, Table
import matplotlib.pyplot as plt
import fnmatch
from scipy.integrate import quad
import pickle 
import math
import mysql
from mysql.connector import Error
import pandas as pd

def load(filename, **kwargs):
    
    """Loads GalaxyCluster object to filename using Pickle"""
    
    with open(filename, 'rb') as fin:
        
        return pickle.load(fin, **kwargs)

moo = clmm.Modeling(massdef = 'mean', delta_mdef = 200, halo_profile_model = 'nfw')

moo.set_cosmo(cosmo)

conn = mysql.connector.connect(host='ccqserv201', user='qsmaster', port=30040)

cursor = conn.cursor(dictionary=True, buffered=True) 

dat_RM = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/Analysis/cosmoDC2/cosmoDC2_details/RedMapper_galaxy_clusters.pkl')

infos_dc2_cut = dat_RM[dat_RM['redshift'] > 0.2]

index = np.arange(len(infos_dc2_cut))

np.random.shuffle(index)

infos_dc2_cut = infos_dc2_cut[index]

print(len(infos_dc2_cut))

for halo in infos_dc2_cut:
    
    halo_id = halo['cluster_id']
    
    file = glob.glob('/sps/lsst/users/cpayerne/cosmoDC2_v1.1.4_image_15_Mpc/cluster_id_RedMapper_*')
    
    name = '/sps/lsst/users/cpayerne/cosmoDC2_v1.1.4_image_15_Mpc/cluster_id_RedMapper_'+ str(halo_id)+'.pkl'
    
    if name in file: 
        print('already loaded')
        continue
    
    z_halo = halo['redshift']
    
    da = cosmo.eval_da(z_halo)
    
    theta_max = ((21./da)) * (180./np.pi)

    ra_cl, dec_cl = halo['ra'], halo['dec']

    query = "SELECT data.coord_ra as ra, data.coord_dec as dec, data.redshift as z_cosmodc2, "
    
    query += "data.mag_i, data.mag_r, data.mag_y, data.galaxy_id as dc2_galaxy_id, "
    
    query += "data.shear_1 as shear1, data.shear_2 as shear2, data.convergence as kappa, "
    
    query += "data.ellipticity_1_true as e1_true, data.ellipticity_2_true as e2_true " 
    
    query += "FROM cosmoDC2_v1_1_4_image.data as data "
    
    query += f"WHERE data.redshift >= {z_halo + 0.1} "
    
    query += f"AND scisql_s2PtInCircle(coord_ra, coord_dec, {ra_cl}, {dec_cl}, {theta_max}) = 1 "
    
    query += f"AND data.mag_i <= 25 "
    
    query += ";"
        
    t0 = time.time()

    tab = pd.read_sql_query(query,conn)

    tf = time.time()
        
    tab = QTable.from_pandas(tab)
    
    tab['g1'], tab['g2'] = clmm.utils.convert_shapes_to_epsilon(tab['shear1'],tab['shear2'],shape_definition = 'shear',kappa = tab['kappa'])
    
    tab['e1_cosmodc2'], tab['e2_cosmodc2'] = clmm.utils.compute_lensed_ellipticity(tab['e1_true'], tab['e2_true'], tab['shear1'], tab['shear2'], tab['kappa'])
    
    norm_e = np.sqrt(tab['e1_cosmodc2']**2 + tab['e2_cosmodc2']**2)
    
    tab['chi1_cosmodc2'], tab['chi2_cosmodc2'] = 2*tab['e1_cosmodc2']/(1 + norm_e**2), 2*tab['e2_cosmodc2']/(1 + norm_e**2)
    
    tab['sigma_c_1_RM'] = moo.eval_critical_surface_density(z_halo, tab['z_cosmodc2'])**(-1.)
    
    dat = clmm.GCData(tab)
    
    cl = clmm.GalaxyCluster('RedMapper_galaxy_cluster', ra_cl, dec_cl, z_halo, dat)  
    
    cl.galcat['host_halo_id'] = halo_id
    
    print('n_saved = ' + str(len(cl.galcat)) + ' in ' + str(tf-t0) + ' (s)')

    cl.save(name)
    
    #cl.save('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/GalaxyClusterCatalogs/test_qserv_gcr/cluster_ID_'+ str(halo_id)+'.pkl')

    print(' ')
    print('------------------')
    print(' ')
