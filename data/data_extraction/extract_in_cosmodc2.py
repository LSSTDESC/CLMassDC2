import sys, os, glob
import numpy as np
from astropy.table import QTable, Table, vstack, join
import pickle 
import mysql
from mysql.connector import Error
import multiprocessing_python
import pandas as pd
import clmm
import healpy
from clmm.galaxycluster import GalaxyCluster
from clmm import Cosmology
import GCRCatalogs

cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
conn = mysql.connector.connect(host='ccqserv201', user='qsmaster', port=30040)
cursor = conn.cursor(dictionary=True, buffered=True)

def query(z, ra, dec, rmax = 10):
    r"""
    Attributes:
    -----------
    z: float
        lens redshift
    ra: float
        lens right ascension
    dec: float
        lens declinaison
    rmax: float
        maximum radius
    """
    da = cosmo.eval_da(z)
    zmax = 3.
    zmin = z + .1
    theta_max = (rmax/da) * (180./np.pi)
    query = "SELECT data.coord_ra as ra, data.coord_dec as dec, data.redshift as z, "
    query += "data.galaxy_id as galaxy_id, "
    query += "data.mag_i, data.mag_r, data.mag_y, "
    query += "data.shear_1 as shear1, data.shear_2 as shear2, data.convergence as kappa, "
    query += "data.ellipticity_1_true as e1_true, data.ellipticity_2_true as e2_true " 
    query += "FROM cosmoDC2_v1_1_4_image.data as data "
    query += f"WHERE data.redshift >= {zmin} AND data.redshift < {zmax} "
    query += f"AND scisql_s2PtInCircle(coord_ra, coord_dec, {ra}, {dec}, {theta_max}) = 1 "
    query += f"AND data.mag_i <= 25 "
    query += ";" 
    return query

def extract_photoz(id_gal, healpix=None, GCRcatalog=None, catalog_name=None):
    r"""
    extract background galaxy catalog with GCRcatalog
    Attributes:
    -----------
    id_gal: array
        background galaxy id
    healpix: array
        list of healpix pixels where to find galaxies
    GCRcatalog: GCRcatalog
        background galaxy GCRcatalog object
    Returns:
    --------
    tab_astropy_ordered: Table
        photoz informations 
    """
    Table_id_gal = Table()
    Table_id_gal['galaxy_id'] = id_gal
    quantities_photoz=['photoz_pdf','photoz_mean','photoz_mode','photoz_odds','galaxy_id']
    dtype = ['object','float', 'float','float','float']
    table_photoz = Table(names = quantities_photoz, dtype = dtype)
    for hp in healpix:
        tab = Table(GCRcatalog.get_quantities(quantities_photoz, 
                                              native_filters=['healpix_pixel=='+str(hp)], 
                                              return_iterator=False))
        mask = np.isin(np.array(tab['galaxy_id']), id_gal)
        print(len(mask[mask==True]))
        tab_cut = tab[mask]
        if n==0: tab_photoz=tab_cut
        else: table_photoz=vstack([table_photoz,tab_cut])
    print(table_photoz)
    n_gal = len(table_photoz['galaxy_id'])
    table_photoz['pzbins'] = np.array([z_bins for i in range(n_gal)])
    table_photoz_ordered = join(table_photoz, Table_id_gal ,keys='galaxy_id')
    return table_photoz_ordered

def extract(lens_redshift=None,lens_ra=None,lens_dec=None, photoz=None, 
            photoz_label='BPZ', GCRcatalog=None):
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
    #qserv
    query_mysql = query(lens_redshift, lens_ra, lens_dec, rmax=5)
    tab = pd.read_sql_query(query_mysql, conn)
    try: 
        tab = QTable.from_pandas(tab)
    except: 
        print('no data')
        return None
    tab['g1'], tab['g2'] = clmm.utils.convert_shapes_to_epsilon(tab['shear1'],tab['shear2'], 
                                                                shape_definition = 'shear',
                                                                kappa = tab['kappa'])
    tab['e1'], tab['e2'] = clmm.utils.compute_lensed_ellipticity(tab['e1_true'], 
                                                                 tab['e2_true'], 
                                                                 tab['shear1'], 
                                                                 tab['shear2'], 
                                                                 tab['kappa'])
    norm_e = np.sqrt(tab['e1']**2 + tab['e2']**2)
    tab['chi1'], tab['chi2'] = 2.*tab['e1']/(1. + norm_e**2), 2.*tab['e2']/(1. + norm_e**2)
    if photoz==True:
        healpix = np.unique(healpy.ang2pix(32, tab['ra'], tab['dec'], nest=False, lonlat=True))
        dat_photoz = extract_photoz(tab['galaxy_id'], healpix, GCRcatalog=GCRcatalog)
        dat_photoz['e1'], dat_photoz['e2'] = tab['e1'], tab['e2']
        w_ls_photoz = compute_galaxy_weights(
                        lens_redshift, cosmo, z_source = None, 
                        shape_component1 = tab['e1'], shape_component2 = tab['e2'], 
                        shape_component1_err = None,
                        shape_component2_err = None, 
                        pzpdf = dat_photoz['photoz_pdf'], pzbins = dat_photoz['pzbins'],
                        use_shape_noise = False, is_deltasigma = True)
    dat = clmm.GCData(tab)
    cl = clmm.GalaxyCluster('GalaxyCluster', lens_ra, lens_dec, lens_redshift, dat)  
    cl.compute_galaxy_weights(z_source='z', pzpdf=None, pzbins=None,
                               shape_component1='e1', shape_component2='e2',
                               shape_component1_err='e1_err', shape_component2_err='e2_err',
                               use_photoz=False, use_shape_noise=False, use_shape_error=False,
                               weight_name='w_ls_true',cosmo=cosmo,
                               is_deltasigma=True, add=True)
    cl.galcat['w_ls_true'] = cl.galcat['w_ls_true']/max(cl.galcat['w_ls_true'])
    if photoz==True: cl.galcat['w_ls' + '_' + photoz_label] = w_ls_photoz/max(w_ls_photoz)
    
    return cl
