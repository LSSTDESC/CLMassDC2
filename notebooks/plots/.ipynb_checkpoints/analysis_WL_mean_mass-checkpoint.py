import numpy as np

analysis_WL = {}
#Modeling_choices

#impact halo profile
analysis_1h_nfw = {'data_path': '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_true.pkl',
                                'halo_profile': 'nfw',
                                'cM': None,
                                'two_halo_term': False,
                                'r_min': 1,
                                'r_max': 5.5,
                                'name_save': '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/halomodel_nfw_freec.pkl',
                                'name_analysis': 'NFW (1h - free concentration)',
                                'ID':'halomodel1'}

#hernquist
analysis_1h_einasto        = analysis_1h_nfw.copy()
analysis_1h_einasto['halo_profile'] = 'Einasto'
analysis_1h_einasto['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/halomodel_einasto_freec.pkl'
analysis_1h_einasto['name_analysis'] = 'Einasto (1h - free concentration)'
analysis_1h_einasto['ID'] = 'halomodel2'

#hernquist
analysis_1h_hernquist        = analysis_1h_nfw .copy()
analysis_1h_hernquist['halo_profile'] = 'Hernquist'
analysis_1h_hernquist['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/halomodel_hernquist_freec.pkl'
analysis_1h_hernquist['name_analysis'] = 'Hernquist (1h - free concentration)'
analysis_1h_hernquist['ID'] = 'halomodel3'

analysis_WL['halo_model'] = [analysis_1h_nfw, analysis_1h_einasto, analysis_1h_hernquist]
###################################################
###################################################
###################################################
###################################################
###################################################
#impact c(M) relation

## Duffy08
analysis_1h_nfw_Duffy08        = analysis_1h_nfw.copy()
analysis_1h_nfw_Duffy08['cM'] = 'Duffy08'
analysis_1h_nfw_Duffy08['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/cM_nfw_Duffy08.pkl'
analysis_1h_nfw_Duffy08['name_analysis'] = 'NFW (1h - Duffy08)'
analysis_1h_nfw_Duffy08['ID'] = 'cM4'

analysis_1h_nfw_Diemer15        = analysis_1h_nfw.copy()
analysis_1h_nfw_Diemer15['cM'] = 'Diemer15'
analysis_1h_nfw_Diemer15['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/cM_nfw_Diemer15.pkl'
analysis_1h_nfw_Diemer15['name_analysis'] = 'NFW (1h - Diemer15)'
analysis_1h_nfw_Diemer15['ID'] = 'cM5'

## Bhattacharya13
analysis_1h_nfw_Bhattacharya13 = analysis_1h_nfw.copy()
analysis_1h_nfw_Bhattacharya13['cM'] = 'Bhattacharya13'
analysis_1h_nfw_Bhattacharya13['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/cM_nfw_Bhattacharya13.pkl'
analysis_1h_nfw_Bhattacharya13['name_analysis'] = 'NFW (1h - Bhattacharya13)'
analysis_1h_nfw_Bhattacharya13['ID'] = 'cM6'

## Duffy08
analysis_1h_nfw_Prada12        = analysis_1h_nfw.copy()
analysis_1h_nfw_Prada12['cM'] = 'Prada12'
analysis_1h_nfw_Prada12['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/cM_nfw_Prada12.pkl'
analysis_1h_nfw_Prada12['name_analysis'] = 'NFW (1h - Prada12)'
analysis_1h_nfw_Prada12['ID'] = 'cM7'

analysis_WL['cM'] = [analysis_1h_nfw_Duffy08, analysis_1h_nfw_Diemer15, analysis_1h_nfw_Bhattacharya13, analysis_1h_nfw_Prada12]
###################################################
###################################################
###################################################
###################################################
###################################################
#NFW + 2h
analysis_2h_nfw        = analysis_1h_nfw_Diemer15.copy()
analysis_2h_nfw['cM'] = 'Diemer15'
analysis_2h_nfw['r_max'] = 10
analysis_2h_nfw['two_halo_term'] = True
analysis_2h_nfw['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/2h_nfw_two_halo_term_Diemer15.pkl'
analysis_2h_nfw['name_analysis'] = 'NFW (1h - Diemer15 + 2h)'
analysis_2h_nfw['ID'] = '2h8'

analysis_2h_nfw_freec        = analysis_1h_nfw_Diemer15.copy()
analysis_2h_nfw_freec['cM'] = None
analysis_2h_nfw_freec['r_max'] = 10
analysis_2h_nfw_freec['two_halo_term'] = True
analysis_2h_nfw_freec['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/2h_nfw_two_halo_term_freec.pkl'
analysis_2h_nfw_freec['name_analysis'] = 'NFW (1h - free concentration + 2h)'
analysis_2h_nfw_freec['ID'] = '2h9'
analysis_WL['two_halo_term'] = [analysis_2h_nfw, analysis_2h_nfw_freec]
###################################################
###################################################
###################################################
###################################################
###################################################
#impact_photoz
analysis_1h_nfw_Diemer15_bpz = analysis_1h_nfw_Diemer15.copy()
analysis_1h_nfw_Diemer15_bpz['data_path'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_BPZ.pkl'
analysis_1h_nfw_Diemer15_bpz['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/PZ_nfw_Diemer15_BPZ.pkl'
analysis_1h_nfw_Diemer15_bpz['name_analysis'] = 'NFW (1h - Diemer15) - BPZ'
analysis_1h_nfw_Diemer15_bpz['ID'] = 'photoz10'

analysis_1h_nfw_Diemer15_flex = analysis_1h_nfw_Diemer15.copy()
analysis_1h_nfw_Diemer15_flex['data_path'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/data_for_notebooks/stacked_esd_profiles_redmapper_flex.pkl'
analysis_1h_nfw_Diemer15_flex['name_save'] = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/WL_mean_masses/PZ_nfw_Diemer15_flex.pkl'
analysis_1h_nfw_Diemer15_flex['name_analysis'] = 'NFW (1h - Diemer15) - FlexZBoost'
analysis_1h_nfw_Diemer15_flex['ID'] = 'photoz11'
analysis_WL['photoz'] = [analysis_1h_nfw_Diemer15_bpz, analysis_1h_nfw_Diemer15_flex]

