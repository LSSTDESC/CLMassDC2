import sys, os
import numpy as np
from astropy.table import QTable, Table, vstack, join
import pickle 
import pandas as pd
import clmm
import healpy
from scipy.integrate import simps
from clmm.galaxycluster import GalaxyCluster
from clmm.dataops import compute_galaxy_weights
from clmm import Cosmology
import utils

r"""
extract background galaxy catalog with qserv for:
cosmodc2:
- true shapes
- true redshift
and GCRCatalogs:
- photoz addons
"""

def extract_photoz(id_gal, healpix=None, GCRcatalog=None):
    r"""
    extract background galaxy catalog with GCRcatalog (healpix subdivision)
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
    quantities_photoz=['photoz_pdf','photoz_mean','photoz_mode',
                       'photoz_odds','redshift','galaxy_id']
    z_bins = GCRcatalog.photoz_pdf_bin_centers
    z_bins[0] = 1e-7
    for n, hp in enumerate(np.array(healpix)):
        tab = GCRcatalog.get_quantities(quantities_photoz, 
                                        native_filters=['healpix_pixel=='+str(hp)])
        tab_astropy = Table()
        tab_astropy['galaxy_id']   = tab['galaxy_id']
        tab_astropy['photoz_pdf']  = tab['photoz_pdf']
        tab_astropy['photoz_mean'] = tab['photoz_mean']
        tab_astropy['photoz_mode'] = tab['photoz_mode']
        tab_astropy['photoz_odds'] = tab['photoz_odds']
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

def extract(lens_redshift=None,
            qserv_query=None, photoz=None, photoz_label='BPZ', GCRcatalog=None, conn_qserv=None,
           cosmo=None):
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
    query_mysql = qserv_query
    tab = pd.read_sql_query(query_mysql, conn_qserv)
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
        #extract photozs
        dat_photoz = extract_photoz(tab['galaxy_id'], healpix, GCRcatalog=GCRcatalog)
    dat = clmm.GCData(tab)
    cl = clmm.GalaxyCluster('GalaxyCluster', 0, 0, 0, dat) 
    #compute weights "true"
#     cl.compute_galaxy_weights(z_source='z', pzpdf=None, pzbins=None,
#                                shape_component1='e1', shape_component2='e2',
#                                shape_component1_err='e1_err', shape_component2_err='e2_err',
#                                use_photoz=False, use_shape_noise=False, use_shape_error=False,
#                                weight_name='w_ls_true',cosmo=cosmo,
#                                is_deltasigma=True, add=True)
    #compute photoz weights
    if photoz==True: 
        sigmac_photoz=utils.compute_photoz_sigmac(lens_redshift,
                              dat_photoz['photoz_pdf'],
                              dat_photoz['pzbins'], cosmo=cosmo)
        cl.galcat['sigmac_photoz_' + '_' + photoz_label]=sigmac_photoz
        colname_to_add=['photoz_odds','photoz_mean','photoz_mode']
        for c in colname_to_add:
            cl.galcat[c+'_'+photoz_label]=dat_photoz[c].data
    return cl
