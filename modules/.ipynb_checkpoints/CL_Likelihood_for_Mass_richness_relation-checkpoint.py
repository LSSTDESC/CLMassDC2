import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import pyccl as ccl
import CL_Mass_richness_relation as scaling_rel
import time
from scipy import interpolate
from astropy.cosmology import FlatLambdaCDM
import CL_WL_mass_conversion as utils
import time
class MR_from_Stacked_Masses():
    r"""
    a class for parametrization of the mass-richness relation, and several likelihoods.
    r"""
    def __init__(self, logm=None, logm_err=None, 
                 richness=None, richness_err=None, 
                 z=None, z_err=None,
                 richness_individual=None, 
                 z_individual=None, 
                 n_cluster_per_bin=None, 
                 weights_individual=None,
                 MRR_object = None):
        r"""data"""
        #stacked
        self.logm=logm
        self.logm_err=logm_err
        self.richness=richness
        self.richness_err=richness_err
        self.z=z
        self.z_err=z_err
        self.modeling = MRR_object
        self.richness_individual=richness_individual
        self.z_individual=z_individual
        self.weights_individual = weights_individual
        self.scaling_rel = MRR_object
    
    def lnLikelihood_binned_classic(self, thetaMC):
        r"""
        Attributes:
        -----------
        thetaMC: array
            free parameters of mass richness relation
        Returns:
        --------
        log likelihood
        """
        logm_mean_expected=self.scaling_rel.lnM(self.richness, self.z, thetaMC)/np.log(10)
        return np.sum(np.log(self.scaling_rel.gaussian(logm_mean_expected,self.logm,self.logm_err)))
    
    def lnLikelihood_individual_zrichness(self, thetaMC):
        r"""
        Attributes:
        -----------
        thetaMC: array
            free parameters of mass richness relation
        Gamma: float
            slope of excess sufrance density
        Returns:
        --------
        log likelihood
        """
        logm_th=[]
        Gamma = self.Gamma
        for i, logm in enumerate(self.logm):
            m_ind=10**self.modeling.lnM(self.richness_individual[i], self.z_individual[i], thetaMC)
            m_G=(m_ind)**Gamma
            m_mean=np.average(m_G, weights=self.weights_individual[i], axis=0)**(1./Gamma)
            logm_th.append(np.log10(m_mean))
        logm_th=np.array(logm_th)
        return np.sum(np.log(self.modeling.gaussian(logm_th,self.logm,self.logm_err)))
    
cosmo_ccl  = ccl.Cosmology(Omega_c=0.265-0.0448, Omega_b=0.0448, h=0.71, A_s=2.1e-9, n_s=0.96, Neff=0, Omega_g=0)    
cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)
deff = ccl.halos.massdef.MassDef(200, 'critical', c_m_relation=None)
concDiemer15 = ccl.halos.concentration.ConcentrationDiemer15(mdef=deff)
concDuffy08 = ccl.halos.concentration.ConcentrationDuffy08(mdef=deff)
concPrada12 = ccl.halos.concentration.ConcentrationPrada12(mdef=deff)
concBhattacharya13 = ccl.halos.concentration.ConcentrationBhattacharya13(mdef=deff)
definition = ccl.halos.massdef.MassDef(200, 'matter', c_m_relation=None)
halobias = ccl.halos.hbias.HaloBiasTinker10(cosmo_ccl, mass_def=definition, mass_def_strict=True)

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

def cM(name):
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
        class ConcDuffy08einasto():
        
            def __init__(self,):
                return None
            
            def _concentration(self, cosmo_ccl, m, a): 
                m_pivot = 2 * 1e12
                A200 = 7.74
                B200 = - 0.123
                C200 = - 0.60
                z = (1/a) - 1
                return A200 * (m/m_pivot) ** B200 * (1 + z) ** C200
        #mc relation
        if   name == 'Diemer15': cmodel = concDiemer15
        elif name == 'Duffy08': cmodel = concDuffy08
        elif name == 'Prada12': cmodel = concPrada12
        elif name == 'Bhattacharya13': cmodel = concBhattacharya13
        elif name == 'Duffy08Einasto': cmodel = ConcDuffy08einasto()
                
        return cmodel

class MR_from_Stacked_ESD_profiles():
    r"""
    a class for parametrization of the mass-richness relation, and several likelihoods.
    r"""
    def __init__(self, richness_individual = None, 
                         z_individual = None, 
                         weights_per_bin_individual = None,
                         covariance_stack = None, 
                         esd_stack = None,
                         radius_stack = None,
                         MRR_object = None, esd_modeling = None, cosmo=None):

        self.richness_individual = richness_individual
        self.z_individual = z_individual
        self.weights_per_bin_individual = weights_per_bin_individual
        self.richness_individual = richness_individual
        self.esd_stack = esd_stack
        self.radius_stack = radius_stack
        self.covariance_stack = covariance_stack
        self.modeling = MRR_object
        self.esd_modeling = esd_modeling
        self.cosmo = cosmo
        
        return None
    
    def reshape_data(self, r_min=1, r_max =5, is_covariance_diagonal = True):
        r"""
        respahe data (profiles and covariances) after selecting radial range
        """
        
        n_stacks = len(self.esd_stack)
        inv_L = []
        esd_stack = []
        radius_stack = []
        covariance_stack = []
        weights = []
        
        for i in range(n_stacks):
            
            w = self.weights_per_bin_individual[i]
            n_per_stack = len(self.z_individual[i])
            index = np.arange(len(self.radius_stack[i]))
            mask_radius = (self.radius_stack[i] > r_min)*(self.radius_stack[i] < r_max)
            index_cut = index[mask_radius]
            esd_stack.append(self.esd_stack[i][mask_radius])
            radius_stack.append(self.radius_stack[i][mask_radius])
            w_cut = np.zeros([n_per_stack, len(index_cut)])
            for h in range(n_per_stack):
                w_cut[h,:] = w[h,:][mask_radius]
            weights.append(w_cut)
            cov_cut = np.array([np.array([self.covariance_stack[i][k,l] for k in index_cut]) for l in index_cut])
            if is_covariance_diagonal == True:
                cov_cut = np.diag(cov_cut.diagonal())
            covariance_stack.append(cov_cut)
            inv_L.append(np.linalg.inv(np.linalg.cholesky(cov_cut)))
        
        self.esd_stack = esd_stack
        self.radius_stack = radius_stack
        self.inv_L = inv_L
        self.covariance_stack = covariance_stack
        self.weights_per_bin_individual = weights
        return None
    
    def halo_regime(self, two_halo = False, esd_2h_nobias_modeling = None, c_m_relation = 'Duffy08'):
        
        if two_halo == True:
            #precompute the unbiased 2h term for all redshift bins
            esd_2h_nobias = np.zeros([len(self.esd_stack), len(self.radius_stack[0])])
            for i in range(len(self.esd_stack)):
                z_mean = np.mean(self.z_individual[i])
                esd_2h_nobias[i,:] = esd_2h_nobias_modeling(self.radius_stack[i], z_mean)
            self.esd_2h_nobias = esd_2h_nobias
            hbias = []
            for i in range(len(self.esd_stack)):
                z_mean = np.mean(self.z_individual[i])
                logm200c_grid = np.linspace(12, 16.5, 50)
                c200c_grid = cM(c_m_relation)._concentration(cosmo_ccl, 10**logm200c_grid, 1./(1. + z_mean))
                m200m = []
                for k in range(len(logm200c_grid)):
                    M200m, c200m = m200m_c200m_from_logm200c_c200c(10**logm200c_grid[k], c200c_grid[k], z_mean)
                    m200m.append(M200m)
                hb = halobias.get_halo_bias(cosmo_ccl, np.array(m200m), 1./(1.+ z_mean), mdef_other = definition)
                def halo_bias_interp(log10m200c):
                    return np.interp(log10m200c, logm200c_grid, hb)
                hbias.append(halo_bias_interp)
            self.halobias_modeling = hbias
            #compute bias function for each redshift
            
    def compute_model(self, cosmo, halo_profile):
        r""" tabulate model """
        logm = np.linspace(13, 16, 25)
        c = np.linspace(1, 10, 20)
        Logm, C = np.meshgrid(logm, c)
        list_model_per_stack = []
        interpolated_model = {}
        tabulated_model = {}
        for i in range(len(self.esd_stack)):
            tabulated_model['stack_'+str(i)] = {}
            interpolated_model['stack_'+str(i)] = {}
            excess_suface_density = np.zeros([len(logm), len(c), len(self.radius_stack[i])])
            z_mean = np.mean(self.z_individual[i])
            for indexm, logmx in enumerate(logm):
                for indexc, cx in enumerate(c):
                    excess_suface_density[indexm,indexc,:] = self.esd_modeling(self.radius_stack[i], logmx, cx, z_mean, cosmo, halo_profile = halo_profile) 
            
            for h, R in enumerate(self.radius_stack[i]): 
                interpolated_model['stack_'+str(i)]['index_R_'+str(h)] = interpolate.interp2d(Logm, C, np.log(excess_suface_density[:,:,h]).T, kind='linear')
                tabulated_model['stack_'+str(i)]['index_R_'+str(h)] = np.log(excess_suface_density[:,:,h])
        
        self.interpolated_model = interpolated_model
        self.tabulated_model = tabulated_model

    def compute_random_gaussian(self):
        
        u1, u2 = [], []
        
        for i in range(len(self.esd_stack)):
                
            u1.append(np.random.randn(len(self.richness_individual[i])))
            u2.append(np.random.randn(len(self.richness_individual[i])))
        self.u1 = u1
        self.u2 = u2
                
        
        
                
    def lnLikelihood(self, thetaMC, which = 'full', scatter_lnc = .2, 
                     c_m_relation = 'Duffy08', halo_profile = 'nfw', 
                     two_halo_term = False, interpolation=False):
        r"""
        Attributes:
        ----------
        thetaMC: array
            log10M0, G, F, sigma_int
        which: str
            which likelihood to use
        scatter_lnc: float
            choose the scatter in log(concentration)
        c_m_relation : str
            which cM relation to use
        Returns:
        --------
        lnL: float
            log-likelihood
        """
        log10M0, G, F, sigma_int = thetaMC
        theta_mean = [log10M0, G, F]
        t = str(time.time()).split('.')[1]
        t = int(t[0])
        np.random.seed(989089*int(abs(np.prod(theta_mean))))
        esd_th_stack = []
        if which == 'full':
            #from Simet et al. 2016 https://arxiv.org/abs/1603.06953
            lnL = 0
            for i in range(len(self.esd_stack)):


                esd_stack_i = self.esd_stack[i]
                radius_i = self.radius_stack[i]

                n_cluster_in_stack = len(self.richness_individual[i])
                
                #u1 = self.u1[i]
                #u2 = self.u2[i]
                u1 = np.random.randn(n_cluster_in_stack)
                u2 = np.random.randn(n_cluster_in_stack)

                ln_mu_m_in_stack = self.modeling.lnM(self.richness_individual[i], self.z_individual[i], theta_mean)
                sigma_int_corrected = np.sqrt(sigma_int**2 + F**2/(self.richness_individual[i]))
                mu_ln_m_in_stack = ln_mu_m_in_stack - (sigma_int_corrected**2)/2
                
                #scatter lambda-M
                lnm_in_stack = mu_ln_m_in_stack + sigma_int_corrected * u1
                c_mu_in_stack = [cM(c_m_relation)._concentration(cosmo_ccl, np.exp(lnm_in_stack)[s], 
                                                                1./(1. + self.z_individual[i][s])) for s in range(len(self.z_individual[i]))]
                #scatter c-M
                lnc_in_stack = np.log(c_mu_in_stack) + scatter_lnc * u2
                c_in_stack = np.exp(lnc_in_stack)
                esd_in_stack = np.zeros([n_cluster_in_stack, len(radius_i)])
                if interpolation==True:
                    interpolation_ = self.interpolated_model['stack_'+str(i)]
                
                def esd_modeling_interpolation(interpolation, logmx, cx):
                    
                    lnds = []
                    for f in range(len(radius_i)):
                        lnds.append(interpolation['index_R_'+str(f)](logmx, cx)[0])
                    return np.exp(np.array(lnds))
                
                for j in range(n_cluster_in_stack):
                    
                    if interpolation==False:
                        esd_in_stack[j,:] = self.esd_modeling(radius_i, lnm_in_stack[j]/np.log(10), c_in_stack[j], 
                                                               self.z_individual[i][j], self.cosmo, 
                                                              halo_profile = halo_profile)
                    elif interpolation==True:
                        logmx, cx = lnm_in_stack[j]/np.log(10), c_in_stack[j]
                        ds = esd_modeling_interpolation(interpolation_, logmx, cx)
                        esd_in_stack[j,:] = ds
                        
                    if two_halo_term == True:
                        hbias_ind = self.halobias_modeling[i](lnm_in_stack[j]/np.log(10))
                        esd_in_stack[j,:] = esd_in_stack[j,:] + hbias_ind * self.esd_2h_nobias[i]
                        
                esd_stack_th = np.average(esd_in_stack, weights = self.weights_per_bin_individual[i], axis=0)

                delta = esd_stack_i-esd_stack_th

                lnL = lnL -.5*(np.sum((self.inv_L[i].dot(delta))**2))
                
                esd_th_stack.append(esd_stack_th)
            
            self.ds = esd_th_stack
            
            self.radius = radius_i
            
            return lnL
        
#         if which == 'simple + no scatter':

#             lnL = 0
#             for i in range(len(self.esd_stack)):

#                 esd_stack_i = self.esd_stack[i]
#                 radius_i = self.radius_stack[i]
#                 ln_mu_m_in_stack = self.modeling.lnM(np.mean(self.richness_individual[i]), 
#                                                      np.mean(self.z_individual[i]), theta_mean) 
#                 mu_m_in_stack = np.exp(ln_mu_m_in_stack)
#                 c_mu_in_stack = cM(c_m_relation)._concentration(cosmo_ccl, mu_m_in_stack, 
#                                                                 1./(1. + np.mean(self.z_individual[i])))
#                 esd_stack_th = self.esd_modeling(np.mean(radius_i, axis=0), np.log10(mu_m_in_stack), 
#                                                  c_mu_in_stack, np.mean(self.z_individual[i]), self.cosmo)
#                 #correction factor
#                 Gamma = .75
#                 sigma_int_corrected = np.sqrt(sigma_int**2 + beta**2/np.mean(self.richness_individual[i]))
#                 mu_ln_m_in_stack = ln_mu_m_in_stack - sigma_int_corrected**2/2
#                 sigma_m_2 = np.exp(2*mu_ln_m_in_stack + sigma_int_corrected**2)*(np.exp(sigma_int_corrected**2)-1)
#                 corr = 1 + .5*Gamma*(Gamma-1)*sigma_m_2/(mu_m_in_stack**2)
#                 delta = esd_stack_i-esd_stack_th#*corr
#                 lnL = lnL -.5*(np.sum((self.inv_L[i].dot(delta))**2))
            
#             return lnL
            
#         if which == 'simple + scatter':
#             lnL = 0
#             for i in range(len(self.esd_stack)):
#                 u1 = np.random.randn(50)
#                 u2 = np.random.randn(50)
#                 esd_stack_i = self.esd_stack[i]
#                 radius_i = self.radius_stack[i]
#                 radius_mean_i =np.mean(radius_i, axis=0)
#                 sigma_int_correct = np.sqrt(sigma_int**2 + beta**2/np.mean(self.richness_individual[i]))
#                 ln_mu_m_in_stack = self.modeling.lnM(np.mean(self.richness_individual[i]), 
#                                                      np.mean(self.z_individual[i]), theta_mean) 
#                 mu_ln_m_in_stack = ln_mu_m_in_stack - sigma_int_correct**2/2
#                 ln_m_in_stack = mu_ln_m_in_stack + sigma_int_correct * u1
#                 m_in_stack = np.exp(ln_m_in_stack)
#                 ln_c_mu_in_stack = np.log(cM(c_m_relation)._concentration(cosmo_ccl, m_in_stack, 
#                                                                           1./(1. + np.mean(self.z_individual[i]))))
#                 ln_c_in_stack = ln_c_mu_in_stack + scatter_lnc * u2
#                 c_in_stack = np.exp(ln_c_in_stack)
#                 esd_table = np.zeros([len(u1), len(self.radius_stack[i])])
                                          
#                 for h in range(len(u1)):
                    
#                     esd_table[h,:] = self.esd_modeling(radius_mean_i, np.log10(m_in_stack[h]), 
#                                                        c_in_stack[h], np.mean(self.z_individual[i]), self.cosmo)
                
#                 delta = esd_stack_i-np.mean(esd_table, axis=0)
#                 lnL = lnL -.5*(np.sum((self.inv_L[i].dot(delta))**2))
            
#             return lnL