import emcee
import numpy as np

def constraint_mcmc(lnL, args, theta0=None, npath=100, nwalkers=200):
    r"""
    mcmc
    """
    ndim=len(theta0)
    pos0=theta0 + 0.01 * np.random.randn(nwalkers, ndim)
    sampler=emcee.EnsembleSampler(nwalkers, ndim, lnL,args=args)
    sampler.run_mcmc(pos0, npath,progress=True)
    sampler=sampler.get_chain(discard = 0, flat = True)
    return sampler

def param_from_chain(chain, n_cut=20):
    
    chain_cut = chain[n_cut:]
    return np.mean(chain_cut, axis=0), np.std(chain_cut, axis=0)