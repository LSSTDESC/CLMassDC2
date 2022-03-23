import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy.optimize import fsolve
import math
from scipy.special import gamma, gammainc

class Modeling():

    def __init__(self, M, concentration, cluster_z, background_density, delta, cosmo):
    
        self.background_density = background_density
        self.M = M
        self.delta = delta
        self.concentration = concentration
        self.cosmo = cosmo
        self.cluster_z = cluster_z
        self.rho_critical = self.cosmo.critical_density(cluster_z).to(u.Msun / u.Mpc**3).value
        """
        alpha : float
            define the overdensity definition alpha * 200 * rho_critical : 1 if 200c , Om(z) if 200m
        """
        if self.background_density == 'mean': self.alpha = self.cosmo.Om(self.cluster_z)
        elif self.background_density == 'critical': self.alpha = 1. 
        self.rdelta = ((3*self.M) / (4*np.pi*self.delta*self.alpha* self.rho_critical)) ** (1./3.)
        self.rs = self.rdelta / self.concentration
        """
        rho_s : float
            the density rho_s of the cluster in M_sun.Mpc^{-3} such as:
            rho(r) = rhos / (r/rs)*(1 + r/rs)**2 
        """
        self.rho_s = (self.delta/3.)*(self.concentration**3/self.delta_c(self.concentration)) * self.alpha * self.rho_critical

    def delta_c(self, c):
        return np.log(1 + c) - c/(1 + c)
    
    def M_in(self,r3d):
        """
        Parameters:
        ----------
        r : float, array
            the 3d radius from the cluster center
        Returns:
        -------
        M : float, array
            the mass within a sphere of radius r (M_sun)
        """
        x = r3d/self.rs
        M_in_r = self.M * self.delta_c(x) / self.delta_c(self.concentration)
        return M_in_r
    
    def density(self,r3d):
        """
        Parameters:
        ----------
        r3d : float
            the distance from the cluster center in Mpc

        Returns:
        -------
        rho : float
            the radial dark matter density of the cluster in M_sun.Mpc^{-3} at radius r
        """
        rho_3d = []
        for R in r3d:
            rho_3d.append(self.rho_s / ((R/self.rs) * (1. + R/self.rs) ** 2))
        return np.array(rho_3d)
