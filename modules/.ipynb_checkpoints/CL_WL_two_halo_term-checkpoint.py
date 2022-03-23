import numpy as np
import pyccl as ccl
from scipy.integrate import quad
from scipy.special import jv
from astropy.cosmology import FlatLambdaCDM
import CL_WL_mass_conversion as conv
cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)
"""
    We compute the excess surface density profile due to the matter distribution of large scale structure surrounding halo;
    Computation of power spectrum and halo bais by Core Cosmology Library
"""

def compute_Pk(kk, z, cosmo_ccl):
    
    r"""
    Atributes:
    ---------
    kk : array, float
        wave vector in Mpc^(-1)
    z : float
        cluster redshift
    cosmo_ccl : cosmology object from ccl
    
    Returns:
    -------
    Pk : The computed linear matter power spectrum for each kk
    """
    
    Pk = []
    
    for k in kk:
        
        Pk.append(ccl.linear_matter_power(cosmo_ccl, k, 1/(1+z)))
        
    return np.array(Pk)

def ds_two_halo_term_unbaised(r, cluster_z, cosmo_ccl, kk, Pk):
    
    r"""
    based on (Oguri & Takada 2011 and Oguri & Hamana 2011, arXiv:1010.0744v2)
    
    Attributes:
    ----------
    r : array, float
        radius from cluster center in Mpc
    cluster_z: float
        cluster redshift
    cosmo_ccl : cosmology object from ccl
    kk, Pk : wavevector and corresponding linear matter power spectrum
    
    Return:
    ------
    the excess surface density induced by large scale struture, divided by the halo bais
    """
    
    Da = ccl.angular_diameter_distance(cosmo_ccl, 1, 1./(1. + cluster_z))
    
    rho_m = ccl.rho_x(cosmo_ccl, 1./(1. + cluster_z), 'matter', is_comoving=False)

    def integrand( l , theta ):
        
        k = l / ( ( 1 + cluster_z ) * Da )
        
        return l * jv( 2 , l * theta ) * np.interp( k , kk , Pk )
    
    two_h = []
    
    for i, R in enumerate(r):
        
        theta = R/Da
    
        val = quad(integrand  , 2*1e-4 , 20000 , args = ( theta ))[0]
        
        two_h.append(val * rho_m / ( 2 * np.pi  * ( 1 + cluster_z )**3 * Da**2 ))
        
    return np.array(two_h)

def s_two_halo_term_unbaised(r, cluster_z, cosmo_ccl, kk, Pk):
    
    r"""
    based on (Oguri & Takada 2011 and Oguri & Hamana 2011, arXiv:1010.0744v2)
    
    Attributes:
    ----------
    r : array, float
        radius from cluster center in Mpc
    cluster_z: float
        cluster redshift
    cosmo_ccl : cosmology object from ccl
    kk, Pk : wavevector and corresponding linear matter power spectrum
    
    Return:
    ------
    the excess surface density induced by large scale struture, divided by the halo bais
    """
    Da = ccl.angular_diameter_distance(cosmo_ccl, 1, 1./(1. + cluster_z))
    rho_m = ccl.rho_x(cosmo_ccl, 1./(1. + cluster_z), 'matter', is_comoving=False)
    def integrand( l , theta ):
        k = l / ( ( 1 + cluster_z ) * Da )
        return l * jv( 0 , l * theta ) * np.interp( k , kk , Pk )
    two_h = []
    for i, R in enumerate(r):
        theta = R/Da
        val = quad(integrand  , 2*1e-4 , 20000 , args = ( theta ))[0]
        two_h.append(val * rho_m / ( 2 * np.pi  * ( 1 + cluster_z )**3 * Da**2 ))
    return np.array(two_h)



def halo_bais(logm = 1, concentration = 1, mdef = 'matter', Delta = 200, halo_def='nfw', cluster_z = 1, cosmo_ccl = 1):
    
    r"""
    Attributes:
    ----------
    logm : float, array
        the log10 of the cluster mass
    mdef : str
        the overdensity mass definition
    Delta : int
        the overdensity relative to mass definition
    cluster_z : float
        the cluster redshift
    cosmo_ccl: object
        Cosmology object of CCL
        
    Returns:
    -------
    hbais : the Tinker (2010) halo bais for mass logm and overdensity definition
    
        (Jeremy L. Tinker et al., arXiv:1001.3162v2)
    r"""
    if mdef == 'matter':
        definition = ccl.halos.massdef.MassDef(Delta, mdef, c_m_relation=None)
        halobais = ccl.halos.hbias.HaloBiasTinker10(cosmo_ccl, mass_def=definition, mass_def_strict=True)
        hbais = halobais.get_halo_bias(cosmo_ccl, 10**logm, 1/(1+cluster_z), mdef_other = definition)
    elif mdef == 'critical':
        if halo_def == 'nfw':
            m200c = 10**logm
            m200m, c200m = conv.M200_to_M200_nfw(M200 = m200c, c200 = concentration, 
                                                 cluster_z = cluster_z, 
                                                 initial = 'critical', final = 'mean', 
                                                 cosmo_astropy = cosmo_astropy)
            definition = ccl.halos.massdef.MassDef(Delta, 'matter', c_m_relation=None)
            halobais = ccl.halos.hbias.HaloBiasTinker10(cosmo_ccl, mass_def=definition, mass_def_strict=True)
            hbais = halobais.get_halo_bias(cosmo_ccl, m200m, 1/(1+cluster_z), mdef_other = definition)
        #elif halo_def == 'einasto':
       #     m200c = 10**logm
        #    m200m, c200m = conv.M200_to_M200_einasto(M200 = m200c, c200 = concentration, 
                                                # cluster_z = cluster_z, 
                                                 #initial = 'critical', final = 'mean', 
                                                # cosmo_astropy = cosmo_astropy)
        #    definition = ccl.halos.massdef.MassDef(Delta, 'matter', c_m_relation=None)
        #    halobais = ccl.halos.hbias.HaloBiasTinker10(cosmo_ccl, mass_def=definition, mass_def_strict=True)
         #   hbais = halobais.get_halo_bias(cosmo_ccl, m200m, 1/(1+cluster_z), mdef_other = definition)
    return hbais
