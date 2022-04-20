import numpy as np
import cosmodc2 as extract_cosmodc2
import lens_data as lens
import matplotlib.pyplot as plt
import GCRCatalogs
import healpy
import pickle,sys

sys.path.append('/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods/modules/')
import edit

import mysql
from mysql.connector import Error
from clmm.dataops import compute_galaxy_weights
from clmm import Cosmology
from scipy.integrate import simps
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)

conn = mysql.connector.connect(host='ccqserv201', user='qsmaster', port=30040)
cursor = conn.cursor(dictionary=True, buffered=True)
start, end = int(sys.argv[1]), int(sys.argv[2])


lens_catalog_name='...'
lens_catalog=edit.load_pickle(lens_catalog_name)
where_to_save='...'
mask=np.arange(start, end)
lens_catalog_truncated=lens_catalog[mask]

gc_flex = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small_with_photozs_flexzboost_v1")
gc_bpz  = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small_with_photozs_v1")
z_bins  = gc_flex.photoz_pdf_bin_centers
z_bins[0] = 1e-7
photoz_label=['bpz', 'flex']
photoz_gc=[gc_bpz, gc_flex]
name_dat_photoz_to_save=['w_ls_photoz', 'photoz_mean', 'photoz_odds', 'photoz_mode']

for n, lens in enumerate(lens_catalog_truncated):
    
    z, ra, dec = lens['z'], lens['ra'], lens['dec']
    #extract with qserv
    bckgd_galaxy_catalog = extract_cosmodc2.extract(lens_redshift=z, 
                                                    lens_ra=ra, lens_dec=dec, rmax=10, 
                                                    conn_qserv=conn, cosmo=cosmo)
    #extract photoz
    id_gal=bckgd_galaxy_catalog.galcat['galaxy_id']
    ras=bckgd_galaxy_catalog.galcat['ra']
    decs=bckgd_galaxy_catalog.galcat['dec']
    
    healpix = np.unique(healpy.ang2pix(32, ras, decs, nest=False, lonlat=True))
    for k, photoz_gc_ in enumerate(photoz_gc):
        #load photoz_data
        dat_photoz=extract_cosmodc2.extract_photoz(id_gal, healpix=healpix, GCRcatalog=photoz_gc_)
        #compute_weights
        w_ls_photoz=extract_cosmodc2.compute_photoz_weights(z, dat_photoz['photoz_pdf'].data, 
                                                            dat_photoz['pzbins'].data,cosmo=cosmo)
        p_background=extract_cosmodc2.compute_background_probability(z, dat_photoz['photoz_pdf'].data, 
                                                            dat_photoz['pzbins'].data)
        
        dat_to_save=[w_ls_photoz, p_background,
                     dat_photoz['photoz_mean'].data,
                     dat_photoz['photoz_odds'].data,
                     dat_photoz['photoz_odds'].data]
        
        #save data
        for s, dat in enumerate(dat_photoz_to_save):
            name=name_dat_photoz_to_save[s]
            bckgd_galaxy_catalog.galcat[name + photoz_label[k]]=dat
            
    edit.save_pickle(where_to_save + name)