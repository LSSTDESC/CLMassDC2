import numpy as np
from scipy.stats import norm

def gaussian(x,mu,sigma):
    r"""
    Attributes:
    -----------
    x: float
        x value
    mu: float
        mean
    sigma: float
        dispersion
    Returns:
    --------
        gaussian(x): float
    """
    return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

def mu_logM_lambda_f(z, logrichness, logM0, alpha, beta, z0, richness0):
    r"""
    z: float
    richness: float
    logM0: float
        normalization
    alpha: float
        slope on redshift
    beta: float
        slope on richness
    z0: float
        pivot redshift
    richness0: float
        pivot richness
    Returns:
    --------
    log likelihood
    """
    return logM0 + alpha*np.log10((1+z)/(1+z0)) + beta*(logrichness-np.log10(richness0))

def lnL_WL_binned(theta, m200c_mean, m200c_err_mean, richness_mean, z_mean, z0, richness0):
    r"""
    Attributes:
    -----------
    theta: array
        free parameters of mass richness relation
    z0: float
        pivot redshift
    richness0: float
        pivot richness
    Returns:
    --------
    log likelihood
    """
    logM0, alpha, beta = theta
    logm_mean_expected = np.array(mu_logM_lambda_f(z_mean, np.log10(richness_mean), logM0, alpha, beta, z0, richness0))
    return np.sum(-.5*(np.log10(m200c_mean)-logm_mean_expected)**2/(m200c_err_mean/m200c_mean)**2)

def lnL_validation_binned(theta, m200c, m200c_rms, logrichness_individual, z_individual, z0, richness0):
    r"""
    Attributes:
    -----------
    theta: array
        free parameters of mass richness relation
    m200c: array
        mean mass
    m200c_rms: array
        error on mean mass
    logrichness_individual: array
        individual richness in bins
    z_individual: array
        individual redshift in bins
    z0: float
        pivot redshift
    richness0: float
        pivot richness
    Returns:
    --------
    log likelihood
    """
    p = []
    logM200c0, alpha, beta = theta
    for i, sample in enumerate(m200c):
            mu_individual = mu_logM_lambda_f(z_individual[i], logrichness_individual[i], logM200c0, alpha, beta, z0, richness0)
            mu_excpected = np.log10(np.average(10**mu_individual))
            err_mu = m200c_rms[i]/(np.log(10)*m200c[i])
            p.append(gaussian(np.log10(m200c[i]), mu_excpected, err_mu))
    return np.sum(np.log(np.array(p)))

def lnL_WL_binned_Gamma(theta, m200c_mean, m200c_err_mean, richness_individual, z_individual, weight_individual, Gamma, z0, richness0):
    r"""
    Attributes:
    -----------
    theta: array
        free parameters of mass richness relation
    m200c_mean: array
        mean mass
    m200c_err_mean: array
        error on mean mass
    richness_individual: array
        individual richness in bins
    z_individual: array
        individual redshift in bins
    weight_individual: array
        average weak lensing weights
    Gamma: array
        slope of excess sufrance density
    z0: float
        pivot redshift
    richness0: float
        pivot richness
    Returns:
    --------
    log likelihood
    """
    p = []
    logM200c0, alpha, beta = theta
    for i, sample in enumerate(m200c_mean):
            logm_individual = mu_logM_lambda_f(z_individual[i], np.log10(richness_individual[i]), logM200c0, alpha, beta, z0, richness0)
            logm_excpected = np.log10(np.average((10**logm_individual)**Gamma, weights = weight_individual[i], axis = 0)**(1/Gamma))
            p.append(norm.pdf(np.log10(m200c_mean[i]), logm_excpected, m200c_err_mean[i]/(np.log(10)*m200c_mean[i])))
    return np.sum(np.log(np.array(p)))
