import sys
import numpy as np
import iminuit
from iminuit import minuit
import os
os.environ['CLMM_MODELING_BACKEND'] = 'nc'
import clmm
from multiprocessing import Pool
from clmm import Cosmology
from astropy.table import Table, QTable, hstack, vstack
import pyccl as ccl
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
cosmo_clmm = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
cosmo_ccl  = ccl.Cosmology(Omega_c=0.265-0.0448, Omega_b=0.0448, h=0.71, A_s=2.1e-9, n_s=0.96, Neff=0, Omega_g=0)
cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)
import emcee
from scipy.integrate import quad

import CL_WL_miscentering as mis
import mcmc
import CL_WL_two_halo_term as twoh
import CL_WL_mass_conversion as utils

#ccl m-c relations
deff = ccl.halos.massdef.MassDef(200, 'critical', c_m_relation=None)
concDiemer15 = ccl.halos.concentration.ConcentrationDiemer15(mdef=deff)
concDuffy08 = ccl.halos.concentration.ConcentrationDuffy08(mdef=deff)
concPrada12 = ccl.halos.concentration.ConcentrationPrada12(mdef=deff)
concBhattacharya13 = ccl.halos.concentration.ConcentrationBhattacharya13(mdef=deff)
definition = ccl.halos.massdef.MassDef(200, 'matter', c_m_relation=None)
halobias = ccl.halos.hbias.HaloBiasTinker10(cosmo_ccl, mass_def=definition, mass_def_strict=True)


class HaloMass_fromStackedProfile():
    r"""a class for the estimation of weak lensing mass from shear profile"""
    def __init__(self, cluster_z, radius, gt, covariance, is_covariance_diagonal = True):
        r"""
        Attributes:
        -----------
        profile: array
            stacked DeltaSigma profile
        covariance: array
            covariance matrix
        """
        self.cluster_z  = cluster_z
        self.ds_obs, self.R = gt, radius
        self.R = np.array([float(r) for r in radius])
        self.is_covariance_diagonal = is_covariance_diagonal
        if  is_covariance_diagonal == False:
            self.cov_ds = covariance
        else:
            self.cov_ds = np.diag(covariance.diagonal())
        self.inv_cov_ds = np.linalg.inv(self.cov_ds)
    
    def set_halo_model(self, halo_model = 'nfw', use_cM_relation = True, 
                       scatter_logm = 0,
                       cM_relation = 'Diemer15', use_two_halo_term = False, scatter_lnc = 0, cosmo = None):
        r"""
        Attributes:
        -----------
        halo_model: str
            halo model
        use_cM_relation: Boolean
            use a cM relation or not
        cM_relation: str
            c-M relation to be used
        use_two_halo_model: Boll
            use 2h term or not
        """
        self.cosmo = cosmo
        self.halo_model = halo_model
        self.scatter_lnc = scatter_lnc
        self.use_two_halo_term = use_two_halo_term
        #clmm 1h-term
        self.use_cM_relation = use_cM_relation
        self.cModel = None
        if self.use_cM_relation == True: 
            cModel = c200c_model(name=cM_relation)
            logm_array = np.linspace(11, 17, 200)
            #tabulated mass-concentration relation
            c_array = cModel._concentration(cosmo_ccl, 10**logm_array, 1./(1. + self.cluster_z))
            def c200c(logm200c): return np.interp(logm200c, logm_array, c_array)
            self.cModel = c200c
            
        #two-halo term
        self.use_two_halo_term = use_two_halo_term
        if self.use_two_halo_term==False:
            self.esd_nobias=None
        else: 
            self.esd_nobias = clmm.theory.compute_excess_surface_density_2h(self.R, self.cluster_z, cosmo_clmm, halobias=1, lsteps=500, validate_input=True)
            logm_array = np.linspace(11, 17, 200)
#             #ccl
            halobias_array = halobias.get_halo_bias(cosmo_ccl, 10**logm_array, 
                                                     1./(1.+ self.cluster_z), 
                                                     mdef_other = definition)
            def hbias(logm200c): return np.interp(logm200c, logm_array, halobias_array)
            self.hbias = hbias
            
    def esd_logm_c_1h(self, R, logm, c):
        mdelta = 10**logm
        delta_mdef=200
        return clmm.compute_excess_surface_density(R, mdelta, c, self.cluster_z, self.cosmo, delta_mdef=200,
                                       halo_profile_model=self.halo_model, massdef='critical')
        
    def esd_1h_term(self, R, logm, c, scatter_lnc,):

        if scatter_lnc == 0:
            return self.esd_logm_c_1h( R, logm, c)
        else: 
            def __integrand__(lnc_, R):
                P_lnc = np.exp(-.5*(lnc_ - np.log(c))**2/(scatter_lnc**2))/(np.sqrt(2*np.pi*scatter_lnc**2))
                return P_lnc * self.esd_logm_c_1h(R, logm, np.exp(lnc_))
            res = []
            for i, r in enumerate(R):
                res.append(quad(__integrand__, -2, 10, args=(r))[0])
            return np.array(res)
    
    def esd_2h_term(self, R, b):
        
        return b * self.esd_nobias
    
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

def c200c_model(name='Diemer15'):
        r"""
        mass-concentration relation for nfw profile
        Attributes:
        -----------
        name: str
            name of mass-concentration used
        Returns:
        --------
        cmodel: ccl concentration object
            selected mass-coencentration relation
        """
        #mc relation
        if name == 'Diemer15': cmodel = concDiemer15
        elif name == 'Duffy08': cmodel = concDuffy08
        elif name == 'Prada12': cmodel = concPrada12
        elif name == 'Bhattacharya13': cmodel = concBhattacharya13
        return cmodel

def reshape_data(radius, gt, covariance, r_min=.1, r_max = 5.5 ):
    r"""
    reshape data according to radial range
    Attributes:
    -----------
    profile: array
        stacked DeltaSigma profile
    covariance: array
        covariance matrix
    """
    R = np.array([float(r) for r in radius])
    mask = (R > r_min)*(R <= r_max)
    index=np.arange(len(R))
    index_cut=index[mask]
    radius_cut = R[index_cut]
    gt_cut=gt[index_cut]
    covariance_cut = np.array([np.array([covariance[i,j] for i in index_cut]) for j in index_cut])
    inv_covariance_cut = np.linalg.inv(covariance_cut)
    return radius_cut, gt_cut, covariance_cut

def m200m_c200m_from_logm200c_c200c(m200c, c200c, z):
        r"""
        conversion
        Attributes:
        -----------
        m200c: float
            cluster mass in 200c convention
        c200c: float
            cluster concentration in 200c convention
        z: float
            cluster redshiftP
        Returns:
        --------
        m200m: folat
            cluster mass in 200m convention
        c200m: folat
            cluster concentration in 200m convention
        """
        m200m, c200m=utils.M_to_M_nfw(m200c, c200c, 200, z, 'critical', 200, 'mean', cosmo_astropy)
        return m200m, c200m
    
def plot_chains(file, name_param = [], name_chain = 'chain'):
    chains = file[name_chain]

    for i, chain in enumerate(chains):
        plt.plot(chain[:,0])
    plt.xlabel('n')
    plt.show()
    

def param_from_chain(file, name_param = [], name_chain = 'chain', n_cut=20):
    
    chains = file[name_chain]
    t_param = {n:[] for n in name_param}
    t_err_param = {n:[] for n in name_param}
    for i, chain in enumerate(chains):
        mean = np.mean(chain[n_cut:], axis=0)
        err = np.std(chain[n_cut:], axis=0)
        for j, n in enumerate(name_param):
            t_param[n].append(mean[j])
            t_err_param[n].append(err[j])
    for j, n in enumerate(name_param):
        file[n] = t_param[n]
        file[n + '_err'] = t_err_param[n]
    return None


def fit_WL_cluster_mass(profile = None, covariance = None, is_covariance_diagonal = True,
                        a = None, b = None, rmax = None, scatter_lnc = .2, scatter_logm=0,
                        two_halo_term = False, fix_c = True, halo_model = 'nfw', 
                        mc_relation='Diemer15', method='minuit'):
    
    r"""fit WL mass from a list of shear profiles and covariance"""
    fit_data_name = ['chain']
    data_to_save = fit_data_name + profile.colnames
    fit_data_name_tot = data_to_save
    tab = {name : [] for name in fit_data_name_tot}
    
    print('fitting...')
    for k, p in enumerate(profile):
        print(str(k)+'/'+str(len(profile)))
        
        cluster_z=p['z_mean']
        radius = p['radius']
        cov = covariance[k]['cov_t']
        gt = p['gt']
        n_in_stack = len(p['redshift'])
        
        radius, gt, cov = reshape_data(radius, gt, cov, r_min=b, r_max = rmax )
        Halo = HaloMass_fromStackedProfile(cluster_z, radius, gt, cov, is_covariance_diagonal = is_covariance_diagonal)
        Halo.gamma = .75
        Halo.scatter_logm = scatter_logm
        Halo.set_halo_model(halo_model = halo_model, 
                            use_cM_relation = fix_c, scatter_lnc = scatter_lnc,
                            cM_relation = mc_relation, 
                            scatter_logm = scatter_logm,
                            use_two_halo_term = two_halo_term)
        Halo.scatter_lnm = Halo.scatter_logm*np.log(10)
        def lnLikelihood(data, inv_cov_data, model):
            "gaussian likelihood"
            delta = data-model
            return -.5 * np.sum( delta * inv_cov_data.dot(delta) )
        print( (1 + .5*Halo.gamma*(Halo.gamma-1)*Halo.scatter_logm**2/n_in_stack))
        def lnL_dm_halo(logm200, c200):
            "full likelihood for logm and concentration"
            if c200 < 0: return -np.inf
            if c200 > 20: return -np.inf
            u=np.random.randn()
            #add scatter in concentration
            lnc200_new = np.log(c200) + u*Halo.scatter_lnc
            c200_new = np.exp(lnc200_new)
            if c200_new < 0: return -np.inf
            if c200_new > 20: return -np.inf
            #add scatter in mean mass
            logm200_new = logm200 + u*Halo.scatter_logm/(n_in_stack**.5)
            if logm200_new<11:
                return -np.inf
            esd_predict = Halo.esd_1h_term(Halo.R, logm200_new, c200_new, 0)
            if Halo.use_two_halo_term == True:
                M200m, c200m = m200m_c200m_from_logm200c_c200c(10**logm200, c200, Halo.cluster_z)
                halobias_val = Halo.hbias(np.log10(M200m))
                esd_predict = esd_predict + Halo.esd_2h_term(Halo.R, halobias_val)
            model = esd_predict
            return lnLikelihood(gt, Halo.inv_cov_ds, esd_predict)
        
        if fix_c == True: nparams = 1
        else: nparams = 2
        
        #mcmc
        #initial position of sampler
        nwalkers = 100
        nstep = 200
        pos =  [14, 6] + 0.01*np.random.randn(nwalkers, 2)
        
        if nparams == 1:
            
            pos_logm = np.array([[pos[:,0][i]] for i in range(len(pos))])
            global lnL_only_mass
            def lnL_only_mass(logm200): 
                "partial likelihood on mass"
                return lnL_dm_halo(logm200[0], Halo.cModel(logm200)[0])
            with Pool() as pool:
                sampler=emcee.EnsembleSampler(nwalkers, 1, lnL_only_mass, pool=pool)
                sampler.run_mcmc(pos_logm, nstep, progress=False)
                    
            sample=sampler.get_chain(discard = 0, flat = True)
        
        if nparams == 2:
            
            global lnL_mass_c
            def lnL_mass_c(p): 
                "full likelihood for mass and concentration"
                logm200, c200 = p
                res = lnL_dm_halo(logm200, c200)
                return res
            
            with Pool() as pool:
                    sampler=emcee.EnsembleSampler(nwalkers, 2, lnL_mass_c, pool=pool)
                    sampler.run_mcmc(pos, nstep, progress=False)
                    
        sample=sampler.get_chain(discard = 0, flat = True)
        
        dat_save = [sample] + [p[s] for s, name in enumerate(profile.colnames)]
        for q, name in enumerate(fit_data_name_tot):
            tab[name].append(dat_save[q])
    return tab
            