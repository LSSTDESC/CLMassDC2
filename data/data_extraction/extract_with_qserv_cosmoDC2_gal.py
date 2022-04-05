import sys, os, glob, time
import numpy as np
from astropy.table import QTable, Table
import pickle 
import mysql
from mysql.connector import Error
import multiprocessing_python
import pandas as pd

import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
from clmm.support import mock_data as mock
import run_extraction as run

cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
moo = clmm.Modeling(massdef = 'mean', delta_mdef = 200, halo_profile_model = 'nfw')
moo.set_cosmo(cosmo)

conn = mysql.connector.connect(host='ccqserv201', user='qsmaster', port=30040)
cursor = conn.cursor(dictionary=True, buffered=True) 
lens_catalog = run.lens_catalog
where_to_save = run.where_to_save
save_key = run.save_key

def extract(n):
    z, ra, dec = lens_catalog[n][run.redshift_name],lens_catalog[n][run.dec_name],lens_catalog[n][run.dec_name]
    index = lens_catalog[n][run.index_name]
    name = where_to_save + save_key + str(index)+'.pkl'
    if name in glob.glob
    da = cosmo.eval_da(z_halo)
    theta_max = ((21./da)) * (180./np.pi)
    query = "SELECT data.coord_ra as ra, data.coord_dec as dec, data.redshift as z_cosmodc2, "
    query += "data.mag_i, data.mag_r, data.mag_y, data.galaxy_id as dc2_galaxy_id, "
    query += "data.shear_1 as shear1, data.shear_2 as shear2, data.convergence as kappa, "
    query += "data.ellipticity_1_true as e1_true, data.ellipticity_2_true as e2_true " 
    query += "FROM cosmoDC2_v1_1_4_image.data as data "
    query += f"WHERE data.redshift >= {z + 0.1} "
    query += f"AND scisql_s2PtInCircle(coord_ra, coord_dec, {ra}, {dec}, {theta_max}) = 1 "
    query += f"AND data.mag_i <= 25 "
    query += ";" 
    tab = pd.read_sql_query(query,conn)
    tab['g1'], tab['g2'] = clmm.utils.convert_shapes_to_epsilon(tab['shear1'],tab['shear2'], 
                                                                shape_definition = 'shear',kappa = tab['kappa'])
    tab['e1_cosmodc2'], tab['e2_cosmodc2'] = clmm.utils.compute_lensed_ellipticity(tab['e1_true'], tab['e2_true'], 
                                                                                   tab['shear1'], tab['shear2'], tab['kappa'])
    norm_e = np.sqrt(tab['e1_cosmodc2']**2 + tab['e2_cosmodc2']**2)
    tab['chi1_cosmodc2'], tab['chi2_cosmodc2'] = 2*tab['e1_cosmodc2']/(1 + norm_e**2), 2*tab['e2_cosmodc2']/(1 + norm_e**2)
    tab['sigma_c_1'] = moo.eval_critical_surface_density(z, tab['z_cosmodc2'])**(-1.)
    dat = clmm.GCData(tab)
    cl = clmm.GalaxyCluster('RedMapper_galaxy_cluster', ra, dec, z, dat)  
    cl.galcat['host_halo_id'] = index
    cl.save(name)
    
save = multiprocessing_python.map(extract, np.arange(len(lens_catalog)), ordered=True)
