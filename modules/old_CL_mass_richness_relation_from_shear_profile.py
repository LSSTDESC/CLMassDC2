import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal


class WL_Mass_Richness():
    r"""
    a class for parametrization of the mass-richness relation, and several likelihoods.
    r"""
    def __init__(self, logm=None, logm_err=None, 
                 richness=None, richness_err=None, 
                 z=None, z_err=None,
                 richness_individual=None, 
                 z_individual=None, 
                 n_cluster_per_bin=None, weights_individual=None):
        r"""data"""
        #stacked
        self.logm=logm
        self.logm_err=logm_err
        self.richness=richness
        self.richness_err=richness_err
        self.z=z
        self.z_err=z_err
        #if n_cluster_per_bin==None:
         #   self.n_cluster_per_bin=np.array([len(z) for z in z_individual])
        #else: self.n_cluster_per_bin=n_cluster_per_bin
        #individual
        self.richness_individual=richness_individual
        self.z_individual=z_individual
        #if weights_individual==None:
         #   self.weights_individual=None
        #else: 
        self.weights_individual = weights_individual
    
    def set_pivot_values(self, z0, richness0):
        r"""pivot values
        Attributes:
        ----------
        z0: float
            pivot redshift
        richness0: float
            pivot richness
        """
        self.z0=z0
        self.richness0=richness0
        
    def gaussian(self, x,mu,sigma):
        r"""
        Attributes:
        -----------
        x: array
            x value
        mu: float
            mean
        sigma: float
            dispersion
        Returns:
        --------
            gaussian: float
        """
        return np.exp(-.5*(x-mu)**2/(sigma**2))/np.sqrt(2*np.pi*sigma**2)

    def mu_logM_lambda(self, richness, z, thetaMC):
        r"""
        z: array
            redshift
        richness: array
            richness
        thetaMC: array
            parameters mass-richness relation
        Returns:
        --------
        mu: array
            mean of mass-richness relation (McClinthock 18)
        """
        logM0, alpha, beta=thetaMC
        return logM0+alpha*np.log10((1.+z)/(1.+self.z0))+beta*(np.log10(richness)-np.log10(self.richness0))
    
    def sigma_mu_logM_lambda(self, richness, z, thetaMC_sigma):
        r"""
        Attributes:
        -----------
        z: array
            redshift
        richness: array
            richness
        thetaMC_sigma: array
            parameters mass-richness relation deviation
        Returns:
        --------
        mu: array
            std of mass-richness relation (McClinthock)
        """
        sigma_logM0, alpha_sigma, beta_sigma=thetaMC_sigma
        return sigma_logM0+alpha_sigma*np.log10((1.+z)/(1.+self.z0))+beta_sigma*(np.log10(richness)-np.log10(self.richness0))

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
        logm_mean_expected=self.mu_logM_lambda(self.richness, self.z, thetaMC)
        return np.sum(np.log(self.gaussian(logm_mean_expected,self.logm,self.logm_err)))
    
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
            m_ind=10**self.mu_logM_lambda(self.richness_individual[i], self.z_individual[i], thetaMC)
            m_G=(m_ind)**Gamma
            #if self.weights_individual==None:
            #    m_mean=np.average(m_G, weights=None, axis=0)**(1./Gamma)
            #else: 
            m_mean=np.average(m_G, weights=self.weights_individual[i], axis=0)**(1./Gamma)
            logm_th.append(np.log10(m_mean))
        logm_th=np.array(logm_th)
        return np.sum(np.log(self.gaussian(logm_th,self.logm,self.logm_err)))
    
    def lnLikelihood_individual_zrichness_random(self, thetaMC):
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
        logM0, alpha, beta, var_int = thetaMC
        if var_int < 0: return -np.inf
        #varlogM_int = 0
        thetamu = logM0, alpha, beta
        for i, logm in enumerate(self.logm):
            logm_ind_mean=self.mu_logM_lambda(self.richness_individual[i], self.z_individual[i], thetamu)
            logm_ind = logm_ind_mean
            m_G=(10**logm_ind)**Gamma
            m_mean=np.average(m_G, weights=self.weights_individual[i]*0+1, axis=0)**(1./Gamma)
            logm_th.append(np.log10(m_mean))
        logm_th=np.array(logm_th)
        return np.sum(np.log(self.gaussian(logm_th + (var_int/self.n_cluster_per_bin)**.5*np.random.randn(),self.logm,self.logm_err)))

    def lnLikelihood_binned_intrinsic_scatter(self,theta):
        r"""
        Attributes:
        -----------
        theta: array
            parameters of mu_logM_lambda + sigma_mu_logM_lambda
        Returns:
        --------
        log likelihood
        """
        logM0, alpha, beta, var_int = theta
        if var_int < 0: return -np.inf
        #varlogM_int = 0
        thetaMC = logM0, alpha, beta
        logm_mean_expected = self.mu_logM_lambda(self.richness, self.z, thetaMC)
        #var = var_WL + var_int (quadratic sum)
        var_logm = self.logm_err ** 2 + (beta**2/(np.log(10)*self.richness) + var_int)/self.n_cluster_per_bin
        return np.sum(np.log(self.gaussian(logm_mean_expected, self.logm, np.sqrt(var_logm))))
    
    def lnLikelihood_individual_masses(self,  logm, richness, z, theta):
        r"""
        likelihood for individual masses, richnesses, and redshifts
        Attributes:
        -----------
        theta: array
            parameters of mu_logM_lambda + sigma_mu_logM_lambda
        Returns:
        --------
        lnL: float
            log-likelihood
        """
        logM0, alpha, beta, sigma_logM0, alpha_sigma, beta_sigma=theta
        thetaMC=logM0, alpha, beta
        thetaMC_sigma=sigma_logM0, alpha_sigma, beta_sigma
        mu=self.mu_logM_lambda(richness, z, thetaMC)
        sigma=self.sigma_mu_logM_lambda(richness, z, thetaMC_sigma)
        return np.sum(np.log(self.gaussian(mu, logm, sigma)))

