import sys
import numpy as np
import iminuit
from iminuit import Minuit
import clmm
from clmm import Cosmology
from astropy.table import Table, QTable, hstack, vstack
import pyccl as ccl
from astropy.cosmology import FlatLambdaCDM
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
cosmo_clmm = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
cosmo_ccl  = ccl.Cosmology(Omega_c=0.265-0.0448, Omega_b=0.0448, h=0.71, A_s=2.1e-9, n_s=0.96, Neff=0, Omega_g=0)
cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)

import CL_WL_miscentering as mis
import CL_WL_two_halo_term as twoh
import CL_WL_mass_conversion as utils

#ccl m-c relations
deff = ccl.halos.massdef.MassDef(200, 'critical', c_m_relation=None)
concDiemer15 = ccl.halos.concentration.ConcentrationDiemer15(mdef=deff)
concDuffy08 = ccl.halos.concentration.ConcentrationDuffy08(mdef=deff)
concPrada12 = ccl.halos.concentration.ConcentrationPrada12(mdef=deff)
concBhattacharya13 = ccl.halos.concentration.ConcentrationBhattacharya13(mdef=deff)

#ccl halo bias
definition = ccl.halos.massdef.MassDef(200, 'matter', c_m_relation=None)
halobais = ccl.halos.hbias.HaloBiasTinker10(cosmo_ccl, mass_def=definition, mass_def_strict=True)

#ccl power spectrum
kk = np.logspace(-5,5 ,100000)

#clmm 1h-term modelling
moo = clmm.Modeling(massdef = 'critical', delta_mdef = 200, halo_profile_model = 'nfw')
moo.set_cosmo(cosmo_clmm)

def c200c_model(name='Diemer15'):
    #mc relation
    if name == 'Diemer15': cmodel = concDiemer15
    elif name == 'Duffy08': cmodel = concDuffy08
    elif name == 'Prada12': cmodel = concPrada12
    elif name == 'Bhattacharya13': cmodel = concBhattacharya13
    return cmodel

def m200m_c200m_from_logm200c_c200c(m200c, c200c, z):
    #mass conversion for 2halo term (halo bias)
    m200m, c200m = utils.M200_to_M200_nfw(M200 = m200c, c200 = c200c, 
                                                cluster_z = z, 
                                                initial = 'critical', final = 'mean', 
                                                cosmo_astropy = cosmo_astropy)
    return m200m, c200m

def chi2_full(logm200, c200, cluster_z, 
         data_R, data_DS, data_cov_DS, data_inv_cov_DS, mask_data_R,
         fix_c, two_halo_term, ds_nobias, is_covariance_diagonal, mc_model = None):
            if fix_c:
                c200 = mc_model._concentration(cosmo_ccl, 10**logm200, 1./(1. + cluster_z))
            moo.set_mass(10**logm200), moo.set_concentration(c200)
            #1h term
            y_predict = np.array([moo.eval_excess_surface_density(data_R[i], cluster_z) for i in range(len(data_R))])
            #2h term
            if two_halo_term == True:
                M200m, c200m = m200m_c200m_from_logm200c_c200c(10**logm200, c200, cluster_z)
                hbais = halobais.get_halo_bias(cosmo_ccl, M200m, 1./(1.+cluster_z), mdef_other = definition)
                y_predict = y_predict + hbais * ds_nobias
            d = (y_predict - data_DS)
            d = np.array([d[i] if mask_data_R[i] == True else 0 for i in range(len(mask_data_R))])
            if is_covariance_diagonal:
                m_2lnL = np.sum((d[mask_data_R]/np.sqrt(data_cov_DS.diagonal()[mask_data_R]))**2)
            else: m_2lnL = np.sum(d*data_inv_cov_DS.dot(d))
            return m_2lnL


def fit_WL_cluster_mass(profile = None, covariance = None, is_covariance_diagonal = True,
                        a = None, b = None, rmax = None, 
                        two_halo_term = False, fix_c = False,halo_model = 'nfw', mc_relation='Diemer15'):

    fit_data_name = ['mask','chi2ndof', 'logm200_w','logm200_w_err', 
                     'c_w', 'c_w_err','1h_term', '2h_term','radius_model']
    data_to_save = fit_data_name + profile.colnames
    fit_data_name_tot = data_to_save
    tab = {name : [] for name in fit_data_name_tot}
    if fix_c == True: 
        c200c_from_logm200c = c200c_model(name=mc_relation)
    else: c200c_from_logm200c = None
    for j, p in enumerate(profile):
        infos_cov = covariance[j]
        cluster_z, ds_obs, cov_ds = p['z_mean'], p['gt'], infos_cov['cov_t']
        inv_cov_ds = np.linalg.inv(cov_ds)
        R = p['radius']
        if two_halo_term == True:
            Pk = ccl.linear_matter_power(cosmo_ccl, kk, 1/(1+cluster_z))
            ds_nobias = twoh.ds_two_halo_term_unbaised(R, cluster_z, cosmo_ccl, kk, Pk)
        else: ds_nobias = None
        rmin, rmax = max(1,a*cluster_z + b) , rmax
        mask = (R > rmin)*(R < rmax)
        
        def chi2(logm200, c200): 
        
            m_2lnL =  chi2_full(logm200, c200, cluster_z, 
                             R, ds_obs, cov_ds, inv_cov_ds, mask,
                             fix_c, two_halo_term, ds_nobias, is_covariance_diagonal, 
                            mc_model = c200c_from_logm200c)
            return m_2lnL
        
        minuit = Minuit(chi2, logm200 = 14, c200 = 4, limit_c200 = (0.01,20), 
                    fix_c200 = fix_c, error_logm200 = .01, error_c200 = .01,
                   limit_logm200 = (12,16), errordef = 1)

        minuit.migrad(),minuit.hesse(),minuit.minos()
        chi2 = minuit.fval/(len(mask[mask == True]) - 1)
        logm_fit = minuit.values['logm200']
        logm_fit_err = minuit.errors['logm200']
        c_fit = minuit.values['c200']
        c_fit_err = minuit.errors['c200']

        if fix_c == True:
            c_fit = c200c_from_logm200c._concentration(cosmo_ccl, 10**logm_fit, 1./(1. + cluster_z))
            c_fit_err = 0
            
        #1h term best fit
        moo.set_mass(10**logm_fit), moo.set_concentration(c_fit)
        ds_1h_term = np.array([moo.eval_excess_surface_density(R[i], cluster_z) for i in range(len(R))])
        #2h term best fit
        if two_halo_term == True:
            M200m, c200m = m200m_c200m_from_logm200c_c200c(10**logm_fit, c_fit, cluster_z)
            hbais = halobais.get_halo_bias(cosmo_ccl, M200m, 1./(1.+cluster_z), mdef_other = definition)
            ds_2h_term = hbais * ds_nobias
        else : ds_2h_term = None
        dat_col_WL = [mask, chi2, logm_fit, logm_fit_err, c_fit, c_fit_err, 
                  ds_1h_term, ds_2h_term, R]
        dat_save = dat_col_WL + [p[s] for s, name in enumerate(profile.colnames)]
        for q, name in enumerate(fit_data_name_tot):
            tab[name].append(dat_save[q])
    return Table(tab)
