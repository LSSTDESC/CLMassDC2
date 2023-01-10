from multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle
from astropy.coordinates import SkyCoord, match_coordinates_3d, match_coordinates_sky
import sys
import emcee
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.table import Table, QTable, hstack, vstack
from astropy import units as u
import corner
from astropy.coordinates import SkyCoord, match_coordinates_3d
cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)
import iminuit
from iminuit import Minuit
cosmo_astropy.critical_density(0.4).to(u.Msun / u.Mpc**3).value

sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMassDC2/modules/')
import CL_WL_miscentering as mis
import analysis_Mass_Richness_relation as analysis
import CL_WL_two_halo_term as twoh
import CL_WL_mass_conversion as utils
import CL_DATAOPS_match_catalogs as match
import CL_WL_DATAOPS_make_profile as prf
#import CL_Likelihood_for_Mass_richness_relation as mass_richness
#import CL_fiducial_mass_richness_relation as fiducial
#import analysis_Mass_richness_relation as analysis


import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
import astropy.units as un
from clmm import Cosmology
from clmm.support import mock_data as mock
import pyccl as ccl

cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)
cosmo_clmm = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
cosmo_ccl  = ccl.Cosmology(Omega_c=0.265-0.0448, Omega_b=0.0448, h=0.71, A_s=2.1e-9, n_s=0.96, Neff=0, Omega_g=0)

def load(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)
ind_profile = np.load('../data/data_new_version/ind_profile_redmapper.pkl', allow_pickle = True)
ind_profile['cluster_id'] = ind_profile['id']
    
Z_bin = analysis.Z_bin
Obs_bin = analysis.Obs_bin
Gamma_5 = 0.75
Gamma_10 = 0.8

profile_sky_stack = prf.stacked_profile(profile = ind_profile,
                    r_in = 'radius_true',
                    gt_in = 'DSt_true', gx_in = 'DSx_true',
                    r_out = 'radius',
                    gt_out = 'gt', gx_out = 'gx',
                    weight = 'W_l_true',
                    z_name = 'redshift', obs_name = 'richness',
                    Z_bin = Z_bin, Obs_bin = Obs_bin,
                    add_columns_to_bin = [ 'W_l_true', 'richness', 'redshift'])

covariance_sky_stack = prf.bootstrap_covariance(profile = ind_profile,
    r_in = 'radius_true',
                    gt_in = 'DSt_true', gx_in = 'DSx_true',
                    r_out = 'radius',
                    gt_out = 'gt', gx_out = 'gx',
                    weight = 'W_l_true',
                    n_boot = 400,
                    z_name = 'redshift', obs_name = 'richness',
                    Z_bin = Z_bin, Obs_bin = Obs_bin)


import CL_Likelihood_for_Mass_richness_relation_v2 as likelihood_mr
import CL_Mass_richness_relation as mr

MR_modeling = mr.WL_Mass_Richness()
MR_modeling.set_pivot_values(analysis.z0, analysis.richness0)

def esd_modeling(R, logm, c, z, cosmo):
    return clmm.compute_excess_surface_density(R, 10**logm, c, z, cosmo, delta_mdef=200,
                                       halo_profile_model='nfw', massdef='critical')

lnL = likelihood_mr.MR_from_Stacked_ESD_profiles(richness_individual = profile_sky_stack['richness'], 
                                                   z_individual = profile_sky_stack['redshift'], 
                                                   weights_per_bin_individual = profile_sky_stack['W_l_true'],
                                                   covariance_stack = covariance_sky_stack['cov_t'], 
                                                   esd_stack = profile_sky_stack['gt'],
                                                   radius_stack = profile_sky_stack['radius'],
                                                   MRR_object = MR_modeling, esd_modeling = esd_modeling, cosmo = cosmo)

lnL.reshape_data(r_min = 1, r_max = 5.5)

def lnLikelihood(p):
    logm0, G, F = p
    ptot = logm0, G, F, .1
    return lnL.lnLikelihood(ptot)

initial_binned = [14, -.2,.75]
npath = 60
ndim = 3
nwalkers = 100
pos_binned = initial_binned + 0.01 * np.random.randn(nwalkers, len(initial_binned))

from multiprocessing import Pool
with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnLikelihood, pool=pool)
        sampler.run_mcmc(pos_binned, npath,progress=True)
sampler_wl = sampler.get_chain(discard = 0, flat = True)