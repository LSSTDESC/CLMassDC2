import numpy as np

analysis_WL = {}
#impact c(M) relation
## Diemer15
analysis_1h_nfw_Diemer15_true = {'data_path': '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_true_test.pkl',
                                'halo_profile': 'nfw',
                                'cM': 'Diemer15',
                                'two_halo_term': False,
                                'interpolation_one_halo_term': True,
                                'r_min': 1,
                                'r_max': 5.5,
                                'W_l_name': 'W_l_true',
                                'name_save': '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/MCMC_chains/nfw_Diemer15_true.pkl'}

## Duffy08
analysis_1h_nfw_Duffy08_true        = analysis_1h_nfw_Diemer15_true.copy()
analysis_1h_nfw_Duffy08_true['cM'] = 'Duffy08'
analysis_1h_nfw_Duffy08_true['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/MCMC_chains/nfw_Duffy08_true.pkl'

## Bhattacharya13
analysis_1h_nfw_Bhattacharya13_true = analysis_1h_nfw_Diemer15_true.copy()
analysis_1h_nfw_Bhattacharya13_true['cM'] = 'Bhattacharya13'
analysis_1h_nfw_Bhattacharya13_true['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/MCMC_chains/nfw_Bhattacharya13_true.pkl'

## Duffy08
analysis_1h_nfw_Prada12_true        = analysis_1h_nfw_Diemer15_true.copy()
analysis_1h_nfw_Prada12_true['cM'] = 'Prada12'
analysis_1h_nfw_Prada12_true['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/MCMC_chains/nfw_Prada12_true.pkl'

analysis_WL['cM'] = [analysis_1h_nfw_Diemer15_true, analysis_1h_nfw_Duffy08_true, analysis_1h_nfw_Bhattacharya13_true, analysis_1h_nfw_Prada12_true]
###########################################################################################################################################################################

#impact of two halo term
analysis_2h_nfw_Diemer15_true = analysis_1h_nfw_Diemer15_true.copy()
analysis_2h_nfw_Diemer15_true['r_max'] = 15
analysis_2h_nfw_Diemer15_true['two_halo_term']=True
analysis_2h_nfw_Diemer15_true['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/MCMC_chains/nfw_Diemer15_true_2h.pkl'

analysis_WL['2h'] = [analysis_2h_nfw_Diemer15_true]
###########################################################################################################################################################################

#impact of two halo term
analysis_1h_einasto_Duffy08_true = analysis_1h_nfw_Diemer15_true.copy()
analysis_1h_einasto_Duffy08_true['cM']= 'Duffy08Einasto'
analysis_1h_einasto_Duffy08_true['halo_profile']= 'einasto'
analysis_1h_einasto_Duffy08_true['interpolation_one_halo_term'] = True
analysis_1h_einasto_Duffy08_true['namesave'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/MCMC_chains/einasto_Duffy08Einasto_true_2h.pkl'

analysis_WL['haloprofile'] = [analysis_1h_einasto_Duffy08_true]
###########################################################################################################################################################################

#impact of photoz
analysis_1h_nfw_Diemer15_bpz = analysis_1h_nfw_Diemer15_true.copy()
analysis_1h_nfw_Diemer15_bpz['W_l_name'] = 'W_l_bpz'
analysis_1h_nfw_Diemer15_bpz['data_path'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_bpz.pkl'
analysis_1h_nfw_Diemer15_bpz['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/MCMC_chains/nfw_Diemer15_bpz.pkl'

analysis_1h_nfw_Diemer15_flex = analysis_1h_nfw_Diemer15_true.copy()
analysis_1h_nfw_Diemer15_flex['W_l_name'] = 'W_l_flex'
analysis_1h_nfw_Diemer15_flex['data_path'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_flex.pkl'
analysis_1h_nfw_Diemer15_flex['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/MCMC_chains/nfw_Diemer15_flex.pkl'

analysis_WL['photoz'] = [analysis_1h_nfw_Diemer15_bpz, analysis_1h_nfw_Diemer15_flex]
###########################################################################################################################################################################


