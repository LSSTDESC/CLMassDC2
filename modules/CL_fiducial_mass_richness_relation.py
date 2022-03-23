import pickle
from astropy.table import Table, QTable, hstack, vstack
import sys
import numpy as np
from iminuit import Minuit
import emcee

import CL_DATAOPS_match_catalogs as match
import CL_Likelihood_for_Mass_richness_relation as mass_richness
import analysis_Mass_Richness_relation as analysis

def load(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)


Z_bin = analysis.Z_bin
Obs_bin = analysis.Obs_bin

dat_RM = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/Galaxy_Cluster_Catalogs_details/cosmoDC2/RedMapper_galaxy_clusters.pkl')
dat_cosmodc2 = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/Galaxy_Cluster_Catalogs_details/cosmoDC2/SkySim5000_DM_halos.pkl')
dat_cosmodc2['M200c'] = dat_cosmodc2['baseDC2/sod_halo_mass']/0.71


def lnL_validation(theta, binned_m200c, binned_m200c_err, logrichness_individual, z_individual):
    return mass_richness.lnL_validation_binned(theta, binned_m200c, binned_m200c_err, logrichness_individual, z_individual, analysis.z0, analysis.richness0)

def constrain_fiducial(used_cluster_id_list = None, low_M_cut = 1e13):

    dat_cosmodc2_cut = dat_cosmodc2[dat_cosmodc2['M200c'] > low_M_cut]
    match_1 = match.match_nearest_neghbor(base_catalog = dat_RM, target_catalog =dat_cosmodc2_cut, label_base = '_RedMapper', label_target = '_cosmoDC2')
    match_1_selection = match.selection_cut(match = match_1, label_base = '_RedMapper', label_target = '_cosmoDC2')
    match_1_selection_repetition = match.find_repetition(match = match_1_selection, label_base = '_RedMapper', label_target = '_cosmoDC2', id_base = 'cluster_id', id_target = 'halo_id', )
    mask = np.isin(match_1_selection_repetition['cluster_id_RedMapper'], used_cluster_id_list)
    match_1_selection_repetition = Table(match_1_selection_repetition)[mask]
    binned_data = match.make_binned(match = match_1_selection_repetition, Z_bin = Z_bin, Richness_bin = Obs_bin)
    m200c_mean_val = np.array(binned_data['m200'])
    m200c_err_mean_val = np.array(binned_data['m200_err'])
    logrichness_mean_val = np.array(binned_data['logrichness'])
    z_mean_val = np.array(binned_data['z_mean'])
    logrichness_individual_val = binned_data['logrichness_in_bin']
    z_individual_val = binned_data['redshift_in_bin']
    npath = 100
    nwalkers = 100
    initial_binned = [14.15,0,0.75]
    pos_binned = initial_binned + 0.01 * np.random.randn(npath, len(initial_binned))
    nwalkers, ndim = pos_binned.shape
    sampler_binned_true = emcee.EnsembleSampler(nwalkers, ndim, lnL_validation, args = (m200c_mean_val, m200c_err_mean_val, logrichness_individual_val, z_individual_val))
    sampler_binned_true.run_mcmc(pos_binned, nwalkers, progress=True)
    flat_sampler_binned_true = sampler_binned_true.get_chain(discard=90, flat=True)
    return np.mean(flat_sampler_binned_true, axis = 0)
