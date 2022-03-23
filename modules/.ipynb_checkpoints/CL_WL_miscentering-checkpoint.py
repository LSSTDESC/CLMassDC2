import numpy as np
from scipy.integrate import quad, simps
import cluster_toolkit as ct

def P_rayleigh(R_off, sigma_off):
    r"""
    Attributes:
    ----------
    R_off : array, float
        Miscentering radius in Mpc
    sigma_off : float
        miscentering scattering in Mpc
    Returns:
    -------
    P_off : the probability density function of the miscentering radius 
    at R_off for the rayleigh distribution
    """
    
    P_off = ( R_off/(sigma_off**2) ) * np.exp(-0.5 * (R_off/sigma_off)**2)
    return P_off

def predict_sigma_excess_miscentering(Sigma, R, R_Sigma, Rmis, cluster_z, kernel, moo):
    r"""
    Attributes:
    ----------
    Sigma : array
        the surface mass density computed for R_Sigma Msun/Mpc^2
    R : array, float
        Radius for DeltaSigma with miscentering to be computed in Mpc
    R_Sigma: array
        radial axis for Sigma in Mpc
    Rmis : float
        the miscentering scattering in Mpc
    cluster_z : float
        the cluster redshift
    kernel : string
        'rayleigh' for the rayleigh distribution
        'gamma' for the gamma distribution
    moo : modelling object of clmm wiht 'cosmo' attribute
    Returns:
    -------
    The miscentered excess surface density relative to Sigma in Msun/Mpc^2
    """
    Omega_m, h = moo.cosmo.get_Omega_m(cluster_z), moo.cosmo['h']
    Sigma_ct_unit = Sigma/((10**12)*h*(1 + cluster_z)**2) #comoving hMsun/Mpc^2
    Rcomoving = h*(1 + cluster_z)*R #comoving Mpc/h
    Rsigma = h*(1 + cluster_z)*R_Sigma #comoving Mpc/h
    Rmis = h*(1 + cluster_z)*Rmis #comoving Mpc/h
    cluster_m_mis, concentration_mis = 3e14, 5 #Msun, no_units
    Sigma_mis = ct.miscentering.Sigma_mis_at_R(Rsigma, Rsigma, Sigma_ct_unit, cluster_m_mis*h, concentration_mis, Omega_m, Rmis, kernel= kernel) #comoving hMsun/Mpc^2
    DeltaSigma_mis = ct.miscentering.DeltaSigma_mis_at_R(Rcomoving,Rsigma, Sigma_mis) #comoving hMsun/Mpc^2
    return DeltaSigma_mis*((10**12)*h*(1 + cluster_z)**2), Sigma_mis*((10**12)*h*(1 + cluster_z)**2) #comoving hMsun/Mpc^2
