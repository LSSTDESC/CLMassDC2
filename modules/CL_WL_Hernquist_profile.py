import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy.optimize import fsolve
import math
from scipy.special import gamma, gammainc

class Modeling():

    def __init__(self, M200, concentration, cluster_z, mass_def, cosmo):
    
        self.mass_def = mass_def
        
        self.M200 = M200
        
        self.concentration = concentration
        
        self.cosmo = cosmo
        
        self.cluster_z = cluster_z
        
        self.rho_critical = self.cosmo.critical_density(self.cluster_z).to(u.Msun / u.Mpc**3).value
        
        """
        alpha : float
            define the overdensity definition alpha * 200 * rho_critical : 1 if 200c , Om(z) if 200m
        """
        
        if self.mass_def == 'mean': self.alpha = self.cosmo.Om(self.cluster_z)
            
        else : self.alpha = 1.
            
        self.r200 = ((self.M200 * 3) / (self.alpha * 800 * np.pi * self.rho_critical)) ** (1./3.)
        
        self.rs = self.r200/self.concentration
    
        """
        rho_s : float
            the density rho_s of the cluster in M_sun.Mpc^{-3} such as:
            rho(r) = rhos / (r/rs)*(1 + r/rs)**3 
        """
        
        self.rho_s = (2/3) * self.concentration * (1 + self.concentration)**2 * self.alpha * self.rho_critical
    
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
        b = (1 + self.concentration)/self.concentration
        
        c = (r3d/self.rs)/(1 + r3d/self.rs)
        
        return self.M200 * (b * c) ** 2
    
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

            rho_3d.append(self.rho_s / ((R/self.rs) * (1. + R/self.rs) ** 3))

        return np.array(rho_3d)
            
            
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    Lensing 
    """
    

    def kappa_u(self,x):

        def inf(x):

            first = 1/(x**2  - 1)
            second = 1 - (2./np.sqrt(1-x**2))*np.arctanh(np.sqrt((1-x)/(1 + x)))

            return first*second

        def equal(x):

            first = 1./3

            return first

        def sup(x):

            first = 1/(x**2 - 1)
            second = 1 - (2./np.sqrt(x**2-1))*np.arctan(np.sqrt((x - 1)/(x + 1)))

            return first*second


        if x > 1:

            return sup(x)

        if x == 1:

            return equal(x)

        if x < 1:

            return inf(x)

    
    def shear_u(self,x):
        
        ## three function for shear

        def ginf(x):

            racine = np.sqrt((1-x)/(1 + x))

            first = 8.*np.arctanh(racine)/(x**2*np.sqrt(1 - x**2))

            second = (4./x**2)*np.log(x/2)

            third = -2./(x**2 - 1)

            fourth = 4.*np.arctanh(racine)/((x**2 - 1)*np.sqrt(1 - x**2))

            return float(first + second + third + fourth)

        def gequal(x):

            first = 10./3 + 4*np.ln(1./2)

            return float(first)

        def gsup(x):

            racine = np.sqrt((x - 1)/(1 + x))

            first = 8.*np.arctan(racine)/(x**2*np.sqrt(x**2 - 1))

            second = (4./x**2)*np.log(x/2)

            third = -2./(x**2 - 1)

            fourth = 4.*np.arctan(racine)/((x**2 - 1)**(3./2))

            return float(first + second + third + fourth)

        
        if x > 1:

            return gsup(x)

        if x < 1:

            return ginf(x)

        if x == 1:

            return gequal(x)

    def critical_density(self, source_z): #Msun/Mpc


        first = self.cosmo.angular_diameter_distance(self.cluster_z).to(u.Mpc)

        second = self.cosmo.angular_diameter_distance(source_z).to(u.Mpc)

        third = self.cosmo.angular_diameter_distance_z1z2(self.cluster_z, source_z).to(u.Mpc)

        fourth = second/(first*third)

        G = const.G.to(u.Mpc**3 / (u.Msun * u.year**2))

        c = const.c.to(u.Mpc / u.year)

        fifth = c**2/(np.pi*4*G)

        return fifth*fourth
    
    def coeff(self, z_source):
        
        return self.rs(self.r200())*self.A(self.concentration)/self.critical_density(z_source).value
    
    def coeff_2(self):
        
        return self.rs(self.r200())*self.A(self.concentration)


    def shear(self, x, z_source): return self.coeff(z_source)*float(self.shear_u(x))


    def kappa(self, x, z_source): return (2) * float(self.kappa_u(x)) * self.coeff(z_source)
    
    
    def deltasigma(self, x): return self.coeff_2() * float(self.shear_u(x))


    def reduced_tangential_shear(self, r, source_z):

        r_200 = self.r200()

        r_s = self.rs(r_200)
        
        y = []
        
        for i, R in enumerate(r):

            x = R/r_s
            
            y.append(self.shear(x, source_z)/(1 - self.kappa(x, source_z)))
            
        return np.array(y)
    
    def excess_surface_density(self, r):
        
        r_200 = self.r200()

        r_s = self.rs(r_200)
        
        y = []
        
        for i, R in enumerate(r):

            x = R/r_s
            
            y.append(self.deltasigma(x))
            
        return np.array(y)            

    





