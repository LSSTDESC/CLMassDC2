import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal


class WL_Mass_Richness():
    r"""
    a class for parametrization of the mass-richness relation, and several likelihoods.
    r"""
    def __init__(self, ):
        r"""data"""
    
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

    def lnM(self, richness, z, thetaMC):
        r"""
        logarithmic mass from McClinthock et al. 2018
        Attributes
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
        log10M0, G, F = thetaMC
        lnM0 = np.log(10)*log10M0
        redshift_evolution = G * np.log((1. + z)/(1. + self.z0))
        richness_evolution = F * np.log(richness/self.richness0)
        return lnM0 + redshift_evolution + richness_evolution
    
    def sigma_lnM(self, richness, z, thetaMC_sigma):
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
        sigma_lnM0, alpha_sigma, beta_sigma=thetaMC_sigma
        return sigma_lnM0+alpha_sigma*np.log((1.+z)/(1.+self.z0))+beta_sigma*np.log(richness/self.richness0)