#import CL_WL_Einasto_profile as ein
import CL_WL_NFW_profile as nfw
#import CL_WL_Hernquist_profile as hernquist
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import fsolve
import numpy as np
import astropy.units as u
from scipy.special import gamma, gammainc


def M200_to_M200_nfw(M200 = 1, c200 = 1, cluster_z = 1, initial = 'critical', final = 'mean', cosmo_astropy = 1):
    r"""
    Attributes:
    ----------
    M200 : array
        the mass M200 of the cluster
    c200 : array
        the concentration c200 associated to mass M200m of the cluster
    z : float
        cluster redshift  
    Returns:
    -------
    M200 : array
        the mass M200 of the cluster
    c200 : array
        the concentration c200 associated to mass M200 of the cluster
    """
    cl_200in = nfw.Modeling(M200, c200, cluster_z, initial, 200, cosmo_astropy)
    M200in, r200in = cl_200in.M, cl_200in.rdelta
    def f(p):
        M200out, c200out = p[0], p[1]
        cl_200out = nfw.Modeling(M200out, c200out, cluster_z, final, 200, cosmo_astropy)
        r200out = cl_200out.rdelta
        """first term"""
        first_term = M200in - cl_200out.M_in(r200in)
        """second term"""
        second_term = M200out - cl_200in.M_in(r200out)
        return first_term, second_term
    x0 = [M200, c200]
    M200out, c200out = fsolve(func = f, x0 = x0)
    return M200out, c200out

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

def M200m_to_M200c_nfw(M200m, c200m, z, cosmo_astropy):
    r"""
    Attributes:
    ----------
    M200m : array
        the mass M200m of the cluster
    c200m : array
        the concentration c200m associated to mass M200m of the cluster
    z : float
        cluster redshift  
    Returns:
    -------
    M200c : array
        the mass M200c of the cluster
    c200c : array
        the concentration c200c associated to mass M200m of the cluster
    """
    cl_200m = nfw.Modeling(M200m, c200m, z, 'mean', 200, cosmo_astropy)
    M200m, r200m = cl_200m.M200, cl_200m.r200
    def f(p):
        M200c, c200c = p[0], p[1]        
        cl_200c = nfw.Modeling(M200c, c200c, z, 'critical',200, cosmo_astropy)       
        r200c = cl_200c.r200      
        """first term"""      
        first_term = M200c - cl_200m.M_in(r200c)      
        """second term"""      
        second_term = M200m - cl_200c.M_in(r200m)     
        return first_term, second_term
    x0 = [M200m, c200m]
    M200c, c200c = fsolve(func = f, x0 = x0)
    return M200c, c200c

def M200_to_M200_einasto(M200 = 1, c200 = 1, cluster_z = 1, initial = 'critical', final = 'mean', cosmo_astropy = 1):
    r"""
    Attributes:
    ----------
    M200 : array
        the mass M200 of the cluster
    c200 : array
        the concentration c200 associated to mass M200m of the cluster
    z : float
        cluster redshift
    Returns:
    -------
    M200 : array
        the mass M200 of the cluster
    c200 : array
        the concentration c200 associated to mass M200 of the cluster
    """
    cl_200in = ein.Modeling(M200, c200, 0.2, cluster_z, initial, cosmo_astropy)
    M200in, r200in = cl_200in.M200, cl_200in.r200
    def f(p):
        M200out, c200out = p[0], p[1]
        cl_200out = ein.Modeling(M200out, c200out, 0.2, cluster_z, final, cosmo_astropy)
        r200out = cl_200out.r200
        """first term"""
        first_term = M200in - cl_200out.M(r200in)
        """second term"""
        second_term = M200out - cl_200in.M(r200out)
        return first_term, second_term
    x0 = [M200, c200]
    M200out, c200out = fsolve(func = f, x0 = x0)
    return M200out, c200out

def M200m_to_M200c_einasto(M200m, c200m, alpha200m, z, cosmo_astropy):
    
    r"""
    Attributes:
    ----------
    M200m : array
        the mass M200m of the cluster
    c200m : array
        the concentration c200m associated to mass M200m of the cluster
    alpha200m : float
        the slope parameter for200m
    z : float
        cluster redshift
        
    Returns:
    -------
    M200c : array
        the mass M200c of the cluster
    c200c : array
        the concentration c200c associated to mass M200m of the cluster
    alpha200c : float
        the slope parameter for 200c
    """
    
    cl_200m = ein.Modeling(M200m, c200m, alpha200m, z, 'mean', cosmo_astropy)
    M200m, c200m, alpha200m = cl_200m.M200, cl_200m.concentration, cl_200m.a
    r200m = cl_200m.r200
    Mtot = cl_200m.Mtot    
    def f(p):    
        logM200c, c200c, alpha200c = p[0], p[1], p[2]   
        M200c = 10**logM200c   
        cl_200c = ein.Modeling(M200c, c200c, alpha200c, z, 'critical', cosmo_astropy)   
        r200c = cl_200c.r200   
        """first_term"""
        first_term = M200m - cl_200c.M(r200m)
        """second_term"""
        second_term = Mtot - cl_200c.Mtot
        """third_term"""
        third_term =  M200c - cl_200m.M(r200c)
        return first_term, second_term, third_term
    x0 = [np.log10(M200m), c200m, alpha200m]
    logM200c, c200c, alpha200c = fsolve(f, x0)
    return 10**logM200c, c200c, alpha200c

def M200m_to_M200c_hernquist(M200m, c200m, z, cosmo_astropy):
    cl_200m = hernquist.Modeling(M200m, c200m, z, 'mean', cosmo_astropy)
    M200m, r200m = cl_200m.M200, cl_200m.r200
    def f(p):  
        M200c, c200c = p[0], p[1]
        cl_200c = hernquist.Modeling(M200c, c200c, z, 'critical', cosmo_astropy)
        r200c = cl_200c.r200
        """first term"""
        first_term = M200c - cl_200m.M(r200c)
        """second term"""
        second_term = M200m - cl_200c.M(r200m)
        return first_term, second_term
    x0 = [M200m, c200m]
    M200c, c200c = fsolve(func = f, x0 = x0)
    return M200c, c200c
