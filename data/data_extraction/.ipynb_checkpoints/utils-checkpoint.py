import sys, os
import numpy as np
from astropy.table import QTable, Table, vstack, join
import pickle 
import pandas as pd
import clmm
import healpy
from scipy.integrate import simps
from clmm.galaxycluster import GalaxyCluster
from clmm.theory import compute_critical_surface_density
from clmm import Cosmology

r"""
extract background galaxy catalog with qserv for:
cosmodc2:
- true shapes
- true redshift
and GCRCatalogs:
- photoz addons
"""
def compute_photoz_sigmac(z_lens, pdf, pzbins, cosmo=None, use_clmm=False):
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
    sigma_c: array
        photometric weak lensing sigma_c
    """
    if pdf.shape!=(len(pdf), len(pzbins)):
        pdf_new=np.zeros([len(pdf), len(pzbins[0])])
        for i, pdf_ in enumerate(pdf):
            pdf_new[i,:] = pdf_
        pdf = pdf_new
    norm=simps(pdf, pzbins, axis=1)
    pdf_norm=(pdf.T*(1./norm)).T
    x=np.linspace(0,0,len(pdf_norm))
    if use_clmm==True:
#         sigmac_2=compute_galaxy_weights(
#                 z_lens, cosmo,z_source = None, 
#                 shape_component1 = x, shape_component2 = x, 
#                 shape_component1_err = None,
#                 shape_component2_err = None, 
#                 pzpdf = pdf_norm, pzbins = pzbins,
#                 use_shape_noise = False, is_deltasigma = True)
        sigma_c=compute_critical_surface_density(cosmo, z_lens, 
                                                 z_source=None, use_pdz=True, 
                                                 pzbins=pzbins, pzpdf=pdf_norm)
        return sigma_c
    else:
        sigmacrit_1 = cosmo.eval_sigma_crit(z_lens, pzbins[0,:])**(-1.)
        sigmacrit_1_integrand = (pdf_norm*sigmacrit_1.T)
        return simps(sigmacrit_1_integrand, pzbins[0,:], axis=1)**(-1.)
    
def compute_p_background(z_lens, pdf, pzbins, use_clmm=False):
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
    p: array
        probability background
    """
    if pdf.shape!=(len(pdf), len(pzbins)):
        pdf_new=np.zeros([len(pdf), len(pzbins[0])])
        for i, pdf_ in enumerate(pdf):
            pdf_new[i,:] = pdf_
        pdf = pdf_new
    norm=simps(pdf, pzbins, axis=1)
    pdf_norm=(pdf.T*(1./norm)).T
    pdf_norm=pdf_norm[:,pzbins[0,:]>=z_lens]
    return simps(pdf_norm, pzbins[0,:][pzbins[0,:]>=z_lens], axis=1)