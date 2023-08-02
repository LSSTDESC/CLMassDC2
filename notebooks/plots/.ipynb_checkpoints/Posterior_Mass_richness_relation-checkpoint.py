import sys
import emcee
import analysis_WL_mean_mass
import pickle
import numpy as np
sys.path.append('/pbs/throng/lsst/users/cpayerne/CLMassDC2/modules/')

import analysis_Mass_Richness_relation as analysis
import CL_Likelihood_for_Mass_richness_relation as mr
import CL_Mass_richness_relation as modeling


def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
    
code, analysisname, index_analysis = sys.argv
analysis_WL_metadata = analysis_WL_mean_mass.analysis_WL[str(analysisname)][int(index_analysis)]

prf = np.load('/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_true.pkl', allow_pickle=True)

    
Gamma_5 = 0.75
Gamma_10 = 0.8
    
file = np.load(analysis_WL_metadata['name_save'], allow_pickle = True)
fits = file['masses']

z0 = analysis.z0
richness0 = analysis.richness0
initial = [14.3,0,1]
npath = 100
ndim=3
nwalkers = 800
pos = initial + 0.01 * np.random.randn(nwalkers, len(initial))
print(analysis_WL_metadata['data_path'])

if analysis_WL_metadata['two_halo_term'] == True:
    if analysis_WL_metadata['data_path'] == '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_true.pkl': 
        Wl_name = 'weight_per_cluster_true_10'
    if analysis_WL_metadata['data_path'] == '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_BPZ.pkl': 
        Wl_name = 'weight_per_cluster_bpz_10'
    if analysis_WL_metadata['data_path'] == '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_flex.pkl': 
        Wl_name = 'weight_per_cluster_flex_10'
    Gamma = Gamma_10
elif analysis_WL_metadata['two_halo_term'] == False:
    if analysis_WL_metadata['data_path'] == '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_true.pkl': 
        Wl_name = 'weight_per_cluster_true_5'
    if analysis_WL_metadata['data_path'] == '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_BPZ.pkl': 
        Wl_name = 'weight_per_cluster_bpz_5'
    if analysis_WL_metadata['data_path'] == '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_flex.pkl': 
        Wl_name = 'weight_per_cluster_flex_5'
    Gamma = Gamma_5
#modeling scaling relation
modeling = modeling.WL_Mass_Richness()
modeling.set_pivot_values( z0, richness0)
#likelihood
fits = fits[fits['z_mean'] < 0.8]
lnL = mr.MR_from_Stacked_Masses(logm=fits['log10M200c_WL'], logm_err=fits['err_log10M200c_WL'], 
         richness=fits['obs_mean'], richness_err=None, 
         z=fits['z_mean'], z_err=None,
         richness_individual=prf['stacked profile']['richness'], 
         z_individual=prf['stacked profile']['redshift'], 
         n_cluster_per_bin=None, weights_individual=prf['stacked profile'][Wl_name], MRR_object=modeling)
lnL.Gamma = Gamma

#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnL.lnLikelihood_individual_zrichness,)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnL.lnLikelihood_binned_classic,)
sampler.run_mcmc(pos, npath,progress=True)
sampler_cut = sampler.get_chain(discard = 0, flat = True)

name = analysis_WL_metadata['name_save'].split('/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/')[1]
path ='/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mass_richness_relation/'
res = {'chains':sampler_cut, 'analysis': analysis_WL_metadata}
name_save = path + name
save_pickle(res, name_save + '_z_lower_than_0.8')

