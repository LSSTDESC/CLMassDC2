import numpy as np
import os 
import analysis_WL_mean_mass
path_mcmc = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/notebooks/plots/Posterior_WL_mean_mass.py'
name_analysis = analysis_WL_mean_mass.analysis_WL.keys()
for name in name_analysis:
    n = len(analysis_WL_mean_mass.analysis_WL[name])
    print(n)
    for j in range(n):
        os.system('python' + ' ' + path_mcmc + ' ' + name + ' ' + str(j))