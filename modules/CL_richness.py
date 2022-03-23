import numpy as np
from astropy.cosmology import FlatLambdaCDM

def mu_loglambda_logM_f(redshift, logm, z0, m0, loglambda0, A_z_mu, A_logm_mu):
    return loglambda0 + A_z_mu * np.log10((1+redshift)/(1 + z0)) + A_logm_mu * (logm-np.log10(m0))

def sigma_loglambda_logm_f(redshift, logm, z0, m0, sigma_lambda0, A_z_sigma, A_logm_sigma):
    return sigma_lambda0 + A_z_sigma * np.log10((1+redshift)/(1 + z0)) + A_logm_sigma * (logm-np.log10(m0))

def lnLambda(redshift, logm, theta_mu, theta_sigma, theta_pivot):
    
    random = np.random.randn(len(logm))
    m0, z0 = theta_pivot
    loglambda0, A_z_mu, A_logm_mu = theta_mu
    sigma_lambda0, A_z_sigma, A_logm_sigma = theta_sigma
    mu = mu_loglambda_logM_f(redshift, logm, z0, m0, loglambda0, A_z_mu, A_logm_mu)
    sigma = sigma_loglambda_logm_f(redshift, logm, z0, m0, sigma_lambda0, A_z_sigma, A_logm_sigma)

    return mu + sigma * random
