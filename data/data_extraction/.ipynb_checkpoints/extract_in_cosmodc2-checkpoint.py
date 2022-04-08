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
from scipy.integrate import simps
from clmm.galaxycluster import GalaxyCluster
from clmm.dataops import compute_galaxy_weights
from clmm import Cosmology

cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
conn = mysql.connector.connect(host='ccqserv201', user='qsmaster', port=30040)
cursor = conn.cursor(dictionary=True, buffered=True)

r"""
extract background galaxy catalog with qserv for:
cosmodc2:
- true shapes
- true redshift
and GCRCatalogs:
- photoz addons
"""
def compute_photoz_weights(z_lens, pdf, pzbins, use_clmm=False):
    r"""
    Attributes:
    -----------
    z_lens: float
        lens redshift
    pdf: array
        photoz distrib
    pzbins: array
        z_bin centers
    Returns:
    --------
    w_ls_photoz: array
        photometric weak lensing weights
    """
    if pdf.shape!=(len(pdf), len(pzbins)):
        pdf_new=np.zeros([len(pdf), len(pzbins[0])])
        for i, pdf_ in enumerate(pdf):
            pdf_new[i,:] = pdf_
        pdf = pdf_new
    norm=simps(pdf, pzbins, axis=1)
    pdf_norm=(pdf.T*(1./norm)).T
    x=np.linspace(0,1,len(pdf_norm))
    if use_clmm==True:
        w=compute_galaxy_weights(
        z_lens, cosmo,z_source = None, 
        shape_component1 = x, shape_component2 = x, 
        shape_component1_err = None,
        shape_component2_err = None, 
        pzpdf = pdf_norm, pzbins = pzbins,
        use_shape_noise = False, is_deltasigma = True)
        return w
    else:
        sigmacrit_2 = cosmo.eval_sigma_crit(z_lens, pzbins[0,:])**(-2.)
        sigmacrit_2_integrand = (pdf_norm*sigmacrit_2.T)
        return simps(sigmacrit_2_integrand, pzbins[0,:], axis=1)

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

def extract_photoz(id_gal, healpix=None, GCRcatalog=None):
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
    quantities_photoz=['photoz_pdf','photoz_mean','photoz_mode','photoz_odds','redshift','galaxy_id']
    z_bins = GCRcatalog.photoz_pdf_bin_centers
    z_bins[0] = 1e-7
    for n, hp in enumerate(healpix):
        tab = GCRcatalog.get_quantities(quantities_photoz, 
                                              native_filters=['healpix_pixel=='+str(hp)], 
                                              return_iterator=False)
        tab_astropy = Table()
        tab_astropy['galaxy_id'] = tab['galaxy_id']
        tab_astropy['photoz_pdf'] =  tab['photoz_pdf']
        tab_astropy['photoz_mean'] =  tab['photoz_mean']
        tab_astropy['photoz_mode'] = tab['photoz_mode']
        tab_astropy['photoz_odds'] = tab['photoz_odds']
        del tab
        mask_id=np.isin(np.array(tab_astropy['galaxy_id']), id_gal)
        tab_astropy=tab_astropy[mask_id]
        if n==0: 
            table_photoz=tab_astropy
        else: 
            tab_=vstack([table_photoz,tab_astropy])
            table_photoz = tab_
    n_gal = len(table_photoz['galaxy_id'])
    table_photoz['pzbins'] = np.array([z_bins for i in range(n_gal)])
    table_photoz_ordered = join(table_photoz, Table_id_gal, keys='galaxy_id')
    return table_photoz_ordered

def extract(lens_redshift=None,lens_ra=None,lens_dec=None, rmax=5, 
            photoz=None, photoz_label='BPZ', GCRcatalog=None):
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
    query_mysql = query(lens_redshift, lens_ra, lens_dec, rmax=rmax)
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
    if photoz==True:
        healpix = np.unique(healpy.ang2pix(32, tab['ra'], tab['dec'], nest=False, lonlat=True))
        dat_photoz = extract_photoz(tab['galaxy_id'], healpix, GCRcatalog=GCRcatalog)
    dat = clmm.GCData(tab)
    cl = clmm.GalaxyCluster('GalaxyCluster', lens_ra, lens_dec, lens_redshift, dat)  
    cl.compute_galaxy_weights(z_source='z', pzpdf=None, pzbins=None,
                               shape_component1='e1', shape_component2='e2',
                               shape_component1_err='e1_err', shape_component2_err='e2_err',
                               use_photoz=False, use_shape_noise=False, use_shape_error=False,
                               weight_name='w_ls_true',cosmo=cosmo,
                               is_deltasigma=True, add=True)
    cl.galcat['w_ls_true'] = cl.galcat['w_ls_true']
    if photoz==True: 
        w_ls_photoz=compute_photoz_weights(lens_redshift,
                              dat_photoz['photoz_pdf'],
                              dat_photoz['pzbins'])
        cl.galcat['w_ls' + '_' + photoz_label]=w_ls_photoz
        cl.galcat['z_mean' + '_' + photoz_label]=dat_photoz['photoz_mean'].data
    return cl
