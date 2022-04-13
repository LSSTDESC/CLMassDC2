#import CL_WL_Einasto_profile as ein
import CL_WL_NFW_profile as nfw
#import CL_WL_Hernquist_profile as hernquist
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import fsolve
import numpy as np
import astropy.units as u

def M_to_M_nfw(M1, c1, delta1, z, rho_bg1, delta2, rho_bg2, cosmo_astropy):
    r"""
    Attributes:
    ----------
    M1 : array
        the mass M of the cluster
    c1 : array
        the concentration c associated to mass M of the cluster
    delta1: float
        overdensity1
    rho_bg1: str
        overdensity definition
    z : float
        cluster redshift 
    delta2: float
        overdensity2
    rho_bgé: str
        overdensity definition2
    Returns:
    -------
    M2 : array
        the mass M2 of the cluster
    c20 : array
        the concentration c2 associated to mass M2 of the cluster
    """
    cl1 = nfw.Modeling(M1, c1, z, rho_bg1, delta1, cosmo_astropy)
    r1 = cl1.rdelta
    def f_to_invert(param):
        M2, c2 = param[0], param[1]        
        cl2 = nfw.Modeling(M2, c2, z, rho_bg2, delta2, cosmo_astropy)
        r2 = cl2.rdelta  
        first_term = M1 - cl2.M_in(r1)
        second_term = M2 - cl1.M_in(r2)  
        return first_term, second_term
    x0 = [M1, c1]
    M2fit, c2fit = fsolve(func = f_to_invert, x0 = x0)
    return M2fit, c2fit