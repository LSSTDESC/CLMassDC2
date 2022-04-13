import numpy as np
from astropy import units as u
from astropy import constants as const
import math
from scipy.special import gamma, gammainc
import pyccl as ccl
from scipy.optimize import fsolve

class Modeling():

    def __init__(self, M200, concentration, slope_parameter, cluster_z, mass_def, cosmo):
    
        self.mass_def = mass_def
        self.M200 = M200
        self.concentration = concentration
        self.cosmo = cosmo
        self.cluster_z = cluster_z
        self.a = slope_parameter
        """
        alpha : float
            define the overdensity definition alpha * 200 * rho_critical : 1 if 200c , Om(z) if 200m
        """
        if self.mass_def == 'mean': self.alpha = self.cosmo.Om(self.cluster_z) 
        else : self.alpha = 1.
        self.rho_critical =  self.cosmo.critical_density(self.cluster_z).to(u.Msun / u.Mpc**3).value
        self.r200 =  ((self.M200 * 3) / (self.alpha * 800 * np.pi * self.rho_critical)) ** (1./3.)
        self.r_2 =  self.r200/self.concentration
        
        """
        rho_2 : float
            the density rho_s of the cluster in M_sun.Mpc^{-3} such as:
            rho(r) = rho-2 * exp[ - 2/\alpha((r/r_2) ** \alpha - 2) ]
        """
        a =  gamma(3./self.a) * gammainc(3/self.a, (2/self.a) * self.concentration ** self.a)
        b = self.alpha * (400./3.) * self.rho_critical
        c = (self.concentration ** 3) * np.exp( - 2./self.a ) * (2/self.a) ** ( (3 - self.a)/self.a )
        self.rho_2 = a ** (-1) * b * c
        self.Mtot = self.M200 * gamma(3./self.a) /(gamma(3./self.a) * gammainc(3/self.a, (2/self.a) * self.concentration ** self.a))
        
    def M(self,r3d):
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
        x = r3d/self.r_2
        up = gamma(3./self.a) * gammainc(3/self.a, (2/self.a) * x ** self.a)
        down = gamma(3./self.a) * gammainc(3/self.a, (2/self.a) * self.concentration ** self.a)
        return self.M200 * (up/down)
        

    def density(self, r3d):
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
        x = r3d/self.r_2     
        return self.rho_2 * np.exp( - (2./self.a) * (x ** self.a - 1.) )

