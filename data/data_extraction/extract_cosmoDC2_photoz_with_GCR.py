import sys
import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins, compute_galaxy_weights,
from clmm.galaxycluster import GalaxyCluster
from clmm import Cosmology
import numpy as np
from astropy.table import Table, vstack, join
import GCRCatalogs
import os
import glob
import pickle
import time
import healpy
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
moo = clmm.Modeling (massdef = 'mean', delta_mdef = 200, halo_profile_model = 'nfw')
moo.set_cosmo(cosmo)

gc_flex = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image_with_photozs_flexzboost_v1")
gc_bpz = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image_with_photozs_v1")
z_bins = gc_flex.photoz_pdf_bin_centers

def extract(photoz):
    r"""
    extract photoz background galaxy catalog
    """
    #find contiguous healpix pixels
    healpix_list = healpy.ang2pix(32, cl['ra'], cl['dec'], nest=False, lonlat=True)
    healpix = np.unique(healpix_list)
    #extract photozs
    for i, hp in enumerate(healpix):
        tab = Table(gc.get_quantities(['photoz_pdf','photoz_mean','photoz_mode','photoz_odds','galaxy_id'],
                                native_filters=['healpix_pixel=='+str(hp)]))
        mask = np.isin(tab['galaxy_id'],np.array(cl['galaxy_id']))
        tab_cut = tab[mask]
        if i == 0:
            tab_astropy = tab_cut
            continue
        tab_astropy = vstack([tab_astropy,tab_cut])
    col_to_rename = ['photoz_pdf','photoz_mean','photoz_mode','photoz_odds']
    for name in col_to_rename:
        tab_astropy.rename_column(name, name + '_' + photoz)
    dat_photozs = clmm.GCData(tab_astropy)
    dat_photozs['pzbins'] = np.array([z_bins for i, ide in enumerate(cl['galaxy_id'])])
    col_default = cl.galcat.colnames
    #join dafault catalog and photoz catalog
    tab_join = join(dat_photozs, cl,keys='galaxy_id')
    cl_full = GalaxyCluster('photoz', ra, dec, z, tab_join)
    #compute weights
    cl_full.compute_galaxy_weights(z, cosmo, z_source=None, pzpdf='photoz_pdf' + '_' + photoz, 
                                   pzbins='pzbins',
                                   shape_component1=None, shape_component2=None,
                                   shape_component1_err=None, shape_component2_err=None,
                                   p_background=None, use_shape_noise=False, is_deltasigma=True,
                                   validate_input=True)
    cl_full_remove = cl_full.galcat
    #remove useless columns
    cl_full_remove.remove_column('pzbins')
    cl_full_remove.remove_column('photoz_pdf_' + photoz)
    cl_full = GalaxyCluster(photoz, ra, dec, z, cl_full_remove)
    return cl_full
