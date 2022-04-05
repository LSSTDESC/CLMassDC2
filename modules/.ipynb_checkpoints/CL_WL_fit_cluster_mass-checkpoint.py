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
import emcee

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
halobias = ccl.halos.hbias.HaloBiasTinker10(cosmo_ccl, mass_def=definition, mass_def_strict=True)

#ccl power spectrum
kk = np.logspace(-5,5 ,100000)

#clmm 1h-term modelling
moo_nfw = clmm.Modeling(massdef = 'critical', delta_mdef = 200, halo_profile_model = 'nfw')
moo_nfw.set_cosmo(cosmo_clmm)
#moo_einasto = clmm.Modeling(massdef = 'critical', delta_mdef = 200, halo_profile_model = 'einasto')
#moo_einasto.set_cosmo(cosmo_clmm)
#moo_hernquist = clmm.Modeling(massdef = 'critical', delta_mdef = 200, halo_profile_model = 'hernquist')
#moo_hernquist.set_cosmo(cosmo_clmm)
moo = moo_nfw
  
class HaloMass_fromStackedProfile():

    def __init__(self, cluster_z, radius, gt, covariance):
        r"""
        Attributes:
        -----------
        profile: array
            stacked DeltaSigma profile
        covariance: array
            covariance matrix
        """
        self.cluster_z  = cluster_z
        self.ds_obs, self.cov_ds, self.R = gt, covariance, radius
        self.R = np.array([float(r) for r in radius])
        self.inv_cov_ds = np.linalg.inv(self.cov_ds)
    
    def set_halo_model(self, halo_model = 'nfw', use_cM_relation = None, cM_relation = 'Diemer15', use_two_halo_term = False):
        r"""
        Attributes:
        -----------
        halo_model: str
            halo model
        use_cM_relation: Boll
            use a cM relation or not
        cM_relation: str
            c-M relation to be used
        use_two_halo_model: Boll
            use 2h term or not
        """
        self.halo_model = halo_model
        #use mass-concentration relation
        self.use_cM_relation = use_cM_relation
        if self.use_cM_relation == False: 
            self.cModel = None
        else: 
            self.cModel = self.c200c_model(name=cM_relation)
            logm_array = np.linspace(11, 17, 200)
            c_array = self.cModel._concentration(cosmo_ccl, 10**logm_array, 1./(1. + self.cluster_z))
            def c200c(logm200c): return np.interp(logm200c, logm_array, c_array)
            self.c200c = c200c
            
        #use two-halo term
        self.use_two_halo_term = use_two_halo_term
        if self.use_two_halo_term == False:
            self.ds_nobias = None
        else: 
            Pk = ccl.linear_matter_power(cosmo_ccl, kk, 1/(1+self.cluster_z))
            self.ds_nobias = twoh.ds_two_halo_term_unbaised(self.R, self.cluster_z, cosmo_ccl, kk, Pk)
            logm_array = np.linspace(11, 17, 200)
            halobias_array = halobias.get_halo_bias(cosmo_ccl, 10**logm_array, 
                                                    1./(1.+ self.cluster_z), 
                                                    mdef_other = definition)
            def hbias(logm200c): return np.interp(logm200c, logm_array, halobias_array)
            self.hbias = hbias
            
            
    def set_radial_range(self, a, b, rmax):
        r"""
        Attributes:
        -----------
        a: float
        b: float
            rmin = (a * cluster_z + b)
        rmax: float
            maximum radius
        r"""
        rmin, rmax = max(1,a * self.cluster_z + b) , rmax
        self.mask_R = (self.R > rmin)*(self.R <= rmax)
        
    def is_covariance_diagonal(self, is_covariance_diag):
        r"""
        Attributes:
        -----------
        is_covariance_diag: Bool
            use diagonal covariance or not
        """
        self.is_covariance_diagonal = is_covariance_diag
        
    def ds_1h_term(self, moo, R):
        res = []
        #for r in R:
            #res.append(moo.eval_excess_surface_density(r, self.cluster_z))
        #return np.array(res)
        return moo.eval_excess_surface_density(R, self.cluster_z)
    def ds_2h_term(self, halobias):
        return halobias * self.ds_nobias
    
    def m200m_c200m_from_logm200c_c200c(self, m200c, c200c, z):
        r"""
        conversion
        Attributes:
        -----------
        m200c: float
            cluster mass in 200c convention
        c200c: float
            cluster concentration in 200c convention
        z: float
            cluster redshift
        Returns:
        --------
        m200m: folat
            cluster mass in 200m convention
        c200m: folat
            cluster concentration in 200m convention
        """
        m200m, c200m = utils.M200_to_M200_nfw(M200 = m200c, c200 = c200c, 
                                                    cluster_z = z, 
                                                    initial = 'critical', final = 'mean', 
                                                    cosmo_astropy = cosmo_astropy)
        return m200m, c200m

    def c200c_model(self, name='Diemer15'):
        r"""
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
        
    def lnL(self, logm200, c200):
        r"""
        Attributes:
        -----------
        logm200: float
            log10 of halo mass
        c200: float
            concentration
        Returns:
        --------
        lnL: log of likelihood
        """
        moo.set_mass(10**logm200), moo.set_concentration(c200)
        y_predict = self.ds_1h_term(moo, self.R)
        if self.use_two_halo_term == True:
            M200m, c200m = self.m200m_c200m_from_logm200c_c200c(10**logm200, c200, self.cluster_z)
            halobias_val = self.hbias(np.log10(M200m))
            y_predict = y_predict + self.ds_2h_term(halobias_val)
        delta = (y_predict - self.ds_obs)
        delta_cut = np.array([delta[s] if is_in == True else 0 for s, is_in in enumerate(self.mask_R)])
        if self.is_covariance_diagonal:
            lnL_val = -.5 * np.sum((delta[self.mask_R]/np.sqrt(self.cov_ds.diagonal()[self.mask_R]))**2)
        else: lnL_val = -.5 * np.sum( delta_cut * self.inv_cov_ds.dot( delta_cut ) )
        return lnL_val
    
    def fit_with_minuit(self, lnL_fit, logm_min, logm_max, c_min, c_max):
    
        if self.use_cM_relation == False:
            def chi2(logm200, c200): return -2 * lnL_fit(logm200, c200)
            minuit = Minuit(chi2, logm200 = 14, c200 = 4)
            minuit.limits=[(logm_min, logm_max),(c_min, c_max)]
            minuit.errordef = 1
            minuit.errors=[0.01,0.01]
            
            minuit.migrad()
            minuit.hesse()
            minuit.minos()
            chi2_val = minuit.fval/(len(self.mask_R[self.mask_R == True]) - 2)
        else: 
            def chi2(logm200): return -2 * lnL_fit(logm200)
            minuit = Minuit(chi2, logm200 = 14)
            minuit = Minuit(chi2, logm200 = 14)
            minuit.limits=[(logm_min, logm_max)]
            minuit.errordef = 1
            minuit.errors=[0.01]
            
            minuit.migrad()
            minuit.hesse()
            minuit.minos()
            chi2_val = minuit.fval/(len(self.mask_R[self.mask_R == True]) - 1)        
        
        logm_fit = minuit.values['logm200']
        logm_fit_err = (minuit.merrors['logm200'].upper - minuit.merrors['logm200'].lower)/2
        #logm_fit_err = minuit.errors['logm200']
        if self.cModel == None:
            c_fit = minuit.values['c200']
            c_fit_err = (minuit.merrors['c200'].upper - minuit.merrors['c200'].lower)/2
            #c_fit_err = minuit.errors['c200']
        else: 
            c_fit = self.c200c(logm_fit)
            c_fit_err = 0
        return logm_fit, logm_fit_err, c_fit, c_fit_err, chi2_val
    
    def fit_with_mcmc(self, lnL_fit, logm_min, logm_max, c_min, c_max):
        
        nwalkers = 100
        nstep = 200
        pos_logm = np.random.randn(nwalkers) * 0.1 + 14
        pos_c = np.random.randn(nwalkers) * 0.00001 + 4
        mask_pos = (logm_max > pos_logm)*(pos_logm > logm_min)*(c_max > pos_c)*(pos_c > c_min)
        pos_logm = pos_logm[mask_pos]
        pos_c = pos_c[mask_pos]
        nwalkers = len(pos_c)
        
        if self.use_cM_relation == True:
            def lnL_fit_mcmc(p):
                logm200 = p
                if logm200 > logm_max: return -np.inf
                if logm200 < logm_min: return -np.inf
                return lnL_fit(logm200)
            pos = pos_logm.T
            ndim = 1
            
        else: 
            def lnL_fit_mcmc(p):
                logm200, c200 = p
                if logm200 > logm_max: return -np.inf
                if logm200 < logm_min: return -np.inf
                if c200 > c_max: return -np.inf
                if c200 < c_min: return -np.inf
                return lnL_fit(logm200, c200)
            pos = np.array([pos_logm, pos_c]).T
            ndim = 2
            
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnL_fit_mcmc)
        sampler.run_mcmc(pos, nstep, progress=False);
        flat_samples = sampler.get_chain(discard=1, thin=3, flat=True)
        mean = np.mean(flat_samples, axis = 0)
        err = np.std(flat_samples, axis = 0)
        if self.use_cM_relation == True:
            return mean[0], err[0], self.c200c(mean), 0, 1
        else:
            return mean[0], err[0], mean[1], err[1], 1, flat_samples
        
    def fit(self, logm_min, logm_max, c_min, c_max, method='minuit'):
        r"""fit the halo mass (and concentration given a method)
        Attributes:
        -----------
        method: str
            method to be used
        Returns:
        -------
        dat_to_save: array
        """
        if self.use_cM_relation == False:
            lnL_fit = self.lnL
        else: 
            def lnL_fit(logm200): 
                c200 = self.c200c(logm200)
                return self.lnL(logm200, c200)
        
        if method == 'minuit':
            res = self.fit_with_minuit(lnL_fit, logm_min, logm_max, c_min, c_max)
            logm_fit, logm_fit_err, c_fit, c_fit_err, chi2_val = res
            chain = 1
        if method == 'mcmc':
            res = self.fit_with_mcmc(lnL_fit, logm_min, logm_max, c_min, c_max)
            logm_fit, logm_fit_err, c_fit, c_fit_err, chi2_val, chain = res

        moo_fit = moo
        moo_fit.set_mass(10**logm_fit), moo_fit.set_concentration(c_fit)
        #compute model:
        ds_1h_term = self.ds_1h_term(moo_fit, self.R)
        if self.use_two_halo_term:
            M200m, c200m = self.m200m_c200m_from_logm200c_c200c(10**logm_fit, c_fit, self.cluster_z)
            halobias_fit = self.hbias(np.log10(M200m))
            ds_2h_term =  self.ds_2h_term(halobias_fit)
        else: ds_2h_term = 0
        dat_to_save =  [self.mask_R, chi2_val, logm_fit, logm_fit_err, c_fit, c_fit_err, 
                  ds_1h_term, ds_2h_term, self.R, chain]
        return dat_to_save

def fit_WL_cluster_mass(profile = None, covariance = None, is_covariance_diagonal = True,
                        a = None, b = None, rmax = None, 
                        two_halo_term = False, fix_c = False,halo_model = 'nfw', mc_relation='Diemer15', method='minuit'):
    fit_data_name = ['mask','chi2ndof', 'logm200_w','logm200_w_err', 
                     'c_w', 'c_w_err','1h_term', '2h_term','radius_model', 'chain']
    data_to_save = fit_data_name + profile.colnames
    fit_data_name_tot = data_to_save
    tab = {name : [] for name in fit_data_name_tot}
    print('fitting...')
    for k, p in enumerate(profile):    
        cluster_z = p['z_mean']
        radius = p['radius']
        cov = covariance[k]['cov_t']
        gt = p['gt']
        Halo = HaloMass_fromStackedProfile(cluster_z, radius, gt, cov)
        Halo.set_halo_model(halo_model = 'nfw', use_cM_relation = fix_c, cM_relation = mc_relation, use_two_halo_term = two_halo_term)
        Halo.set_radial_range(a, b, rmax)
        Halo.is_covariance_diagonal(is_covariance_diagonal)
        logm_min, logm_max, c_min, c_max = 11, 17, .01, 20
        data_fit_WL = Halo.fit(logm_min, logm_max, c_min, c_max, method=method)
        dat_save = data_fit_WL + [p[s] for s, name in enumerate(profile.colnames)]
        for q, name in enumerate(fit_data_name_tot):
            tab[name].append(dat_save[q])
    return Table(tab)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
r"""
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
    

    Attributes:
    -----------
    profile: Table
        stacked profile table
    covariance: Table
        stacked coavriance table
    is_covariance_diagonal: boolean
        use diagonal_covariance or not
    a: float
        rmin = a + b*z
    b: float
        rmin = a + b*z
    rmax: float
        maximum radius
    two_halo_term: boolean
        use 2h term or not
    fix_c: boolean
        fic concentration to m-c relation of not
    halo_model: str
        halo model
    mc_relation: str
        name of mass-concentration used
    Returns:
    --------
    fit: Table

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
        cluster_z, ds_obs, cov_ds, R = p['z_mean'], p['gt'], infos_cov['cov_t'], p['radius']
        inv_cov_ds = np.linalg.inv(cov_ds)
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
"""